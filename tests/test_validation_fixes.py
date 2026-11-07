"""Regression tests for the 2026-11-07 validation review fixes."""
import unittest.mock
import copy
import unittest
import numpy as np
import torch
from kiosk_adaptive.design import make_court, Objective, lhs_objectives
from kiosk_adaptive.environment import simulate
from kiosk_adaptive.inventory import Inventory, demand_stock
from kiosk_adaptive.models import Settings
from kiosk_adaptive.policy import ActorCritic, RuleController, targets, INDEX
from kiosk_adaptive.study import period, period_score, evaluate, summarize, combine_replicas, baseline_audit, DEFAULT, visitor_mean


class ValidationFixTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.s = Settings()
        self.waste = Objective('waste', 1/3, 1/3, 1/3, 0., 0.)
        self.mixed = Objective('mixed', .4, .3, .3, .5, .7)

    def test_closed_days_score_zero_and_waste_moves_to_service_day(self):
        court = make_court(self.s, 5, 'test', 150, 6)
        rows, _ = period(court, self.s, self.waste, 14, 'closed', 150)
        self.assertTrue(any(not r['opened'] for r in rows))
        for r in rows:
            if not r['opened']:
                self.assertEqual(r['score'], 0.); self.assertEqual(r['waste_cost'], 0.)
            # No placeholder denominator: bounded per-visitor scores.
            self.assertLess(abs(r['score']), 5.)
        charged = sum(r['waste_cost'] for r in rows)+rows[-1].get('unattributed_waste_cost', 0.)
        self.assertAlmostEqual(charged, sum(r['expired_today'] for r in rows), places=6)
        self.assertAlmostEqual(period_score(rows), np.mean([r['score'] for r in rows if r['opened']]))

    def test_demand_sized_stock_and_censored_forecast(self):
        court = make_court(self.s, 5, 'test', 150, 6)
        stock = demand_stock(court, 100, self.s)
        capacity = court.initial_stock
        self.assertLess(sum(sum(r.values()) for r in stock), sum(sum(r.values()) for r in capacity))
        court.initial_stock = stock
        a, b = Inventory(court, 1, 'x'), Inventory(court, 1, 'x')
        used = [{k: 1. for k in r} for r in stock]
        lost = [{k: 50. for k in r} for r in stock]
        before = copy.deepcopy(a.forecast)
        a.finish_day(0, used, a.stock(), lost=None)
        b.finish_day(0, used, b.stock(), lost=lost)
        for fa, fb, f0 in zip(a.forecast, b.forecast, before):
            for k in fa:
                self.assertGreater(fb[k], fa[k])
                self.assertGreaterEqual(fb[k], f0[k])

    def test_usable_stock_excludes_batches_expiring_before_sale(self):
        court = make_court(self.s, 0, 'test', 50)
        inv = Inventory(court, 1, 'x')
        k = 'protein'
        inv.batches[0] = [dict(ingredient=k, quantity=1000., expires=1)]
        self.assertEqual(inv.usable(0, k, 0), 0.)
        inv.batches[0] = [dict(ingredient=k, quantity=1000., expires=3)]
        self.assertAlmostEqual(inv.usable(0, k, 0), 2*inv.forecast[0][k])

    def test_venue_lives_outlast_lead_time(self):
        for split in ('train', 'validation', 'test'):
            for i in range(6):
                p = make_court(self.s, i, split, 50, 6).provenance['parameters']
                for key in ('protein_life_days', 'vegetable_life_days', 'staple_life_days'):
                    self.assertGreater(p[key], p['lead_days'])

    def test_policy_sees_only_enterable_inputs(self):
        court = make_court(self.s, 0, 'test', 80)
        seen = []
        def choose(x, rng):
            seen.append(x.copy()); return 0
        simulate(court, self.s, 'neural', visitors=80, controller=choose)
        x = np.concatenate(seen)
        self.assertTrue(np.allclose((x[:, INDEX['hurry_input']]*4) % 1, 0, atol=1e-5))
        self.assertTrue(np.allclose(np.round(x[:, INDEX['budget_input']]*11000) % 1000, 0, atol=1))
        # Changing the hidden class gap must not change any observation.
        import kiosk_adaptive.customers as customers
        real = customers.generate_customers
        def shifted(*args, **kwargs):
            out = real(*args, **kwargs)
            for c in out: c['class_gap'] += 30.
            return out
        again = []
        customers_patch = unittest.mock.patch('kiosk_adaptive.environment.generate_customers', side_effect=shifted)
        with customers_patch:
            simulate(court, self.s, 'neural', visitors=80, controller=lambda x, rng: again.append(x.copy()) or 0)
        self.assertTrue(np.array_equal(seen[0], again[0]))

    def test_missing_inputs_are_neutral(self):
        s = Settings(input_missing_probability=1.)
        court = make_court(s, 0, 'test', 40)
        seen = []
        simulate(court, s, 'neural', visitors=40, controller=lambda x, rng: seen.append(x.copy()) or 0)
        x = np.concatenate(seen)
        self.assertTrue(np.allclose(x[:, [0, 1, 2, 4, 5, 6]], .5))

    def test_reward_sums_equal_daily_scores(self):
        court = make_court(self.s, 1, 'test', 60)
        for mode in ('decomposed', 'daily'):
            rows, samples = period(court, self.s, self.mixed, 5, 'reward', 60, ActorCritic(16), learn=True, reward_mode=mode)
            self.assertAlmostEqual(sum(s['reward'] for s in samples), sum(r['score'] for r in rows), places=6)
            if mode == 'decomposed':
                self.assertGreater(len({round(s['reward'], 9) for s in samples}), 2)

    def test_gae_one_matches_monte_carlo(self):
        samples = [dict(reward=r, value=v) for r, v in [(.1, .5), (0., .2), (.3, -.1), (-.2, .4)]]
        a, b = targets(samples, None), targets(samples, 1.)
        torch.testing.assert_close(a[0], b[0]); torch.testing.assert_close(a[1], b[1])
        c = targets(samples, 0.)
        self.assertAlmostEqual(float(c[1][0]), .1+.2-.5, places=6)

    def test_rule_controllers(self):
        x = torch.zeros(3, 49)
        x[:, INDEX['taste_input']] = .5
        x[1, INDEX['dish_taste']] = .5; x[0, INDEX['dish_taste']] = 1.; x[2, INDEX['dish_taste']] = 0.
        x[:, INDEX['eta']] = torch.tensor([.1, .9, .5]); x[:, INDEX['margin']] = torch.tensor([.2, .1, .9])
        x[:, INDEX['budget_input']] = 1.; x[:, INDEX['price']] = .3
        best = lambda kind, o=None: int(RuleController(kind, o)(x)[0].probs.argmax())
        self.assertEqual(best('rule_fit'), 1); self.assertEqual(best('rule_fast'), 0); self.assertEqual(best('rule_margin'), 2)
        self.assertEqual(best('rule_objective', Objective('t', 1., 0., 0., 1., 0.)), 1)

    def test_controls_only_in_test_and_interval_summary(self):
        cfg = dict(DEFAULT, evaluation_repeats=1, test_courts=2, validation_courts=1, period_days=2, visitors=10,
                   demand_mode='fixed', behaviour_worlds=['mixed_logit'])
        v = evaluate(ActorCritic(8), self.s, self.mixed, cfg, 'validation')
        self.assertTrue(all(p['controls'] == {} for p in v))
        t = evaluate(ActorCritic(8), self.s, self.mixed, cfg, 'test')
        result = summarize(t)
        stable = result['stable']
        self.assertEqual(stable['courts'], 2)
        self.assertEqual(len(stable['t95']), 2); self.assertEqual(len(stable['bootstrap95']), 2)
        self.assertIn(stable['learning_check']['verdict'], ('ai_beats_best_rule', 'ai_not_better_than_best_rule'))
        summaries = [dict(objective=dict(id='mixed'), results=result), dict(objective=dict(id='mixed'), results=result)]
        combined = combine_replicas(summaries)['mixed']
        self.assertEqual(combined['replicas'], 2)
        self.assertAlmostEqual(combined['stable']['mean'], stable['mean_delta'])

    def test_baseline_audit_flags(self):
        cfg = dict(DEFAULT, period_days=7, visitors=150)
        court = make_court(self.s, 0, 'test', 150, 2)
        report = baseline_audit({'test': [court]}, self.s, cfg, self.mixed,
                                dict(DEFAULT['audit_thresholds'], min_served_fraction=1.01))
        self.assertIn('low_service', report['courts'][court.id]['flags'])
        self.assertEqual(report['flagged'], [court.id])


if __name__ == '__main__': unittest.main()
