import copy
from dataclasses import replace
import unittest
from kiosk_adaptive.models import Settings
from kiosk_adaptive.design import make_court
from kiosk_adaptive.venues import evidence,coverage,TYPES,UNSUPPORTED
from kiosk_adaptive.inventory import Inventory
from kiosk_adaptive.study import period,visitor_mean


class VenueTests(unittest.TestCase):
    def test_joint_case_anchors_and_missing_fields(self):
        s=Settings()
        courts=[make_court(s,i,'train',10,8) for i in range(8)]
        ev={c['id']:c for c in evidence()['cases']}
        for c in courts[:4]:
            a=ev[c.provenance['case_ids'][0]]
            self.assertEqual(len(c.servers),a['vendors']);self.assertEqual(c.seats,a['seats'])
            if 'hours' in a:self.assertEqual(c.duration,60*a['hours'])
            if 'kiosks' in a:self.assertEqual(c.kiosks,a['kiosks'])
        self.assertEqual([(len(c.servers),c.seats) for c in courts[:2]],[(2,128),(3,217)])
        self.assertEqual(coverage(courts)['types']['campus']['input_lhs_strata'],True)
        self.assertTrue(all(d.name.startswith('가상 ') and not d.observed_name for c in courts for d in c.dishes))
        self.assertTrue(all(not set(c.provenance['parameters'])&set(UNSUPPORTED) for c in courts))

    def test_types_planned_exclusion_and_capacity_coupling(self):
        s=Settings(venue_types=TYPES)
        courts=[make_court(s,i,'test',10,40) for i in range(40)]
        result=coverage(courts)['types']
        self.assertEqual(set(result),set(TYPES))
        for kind,stats in result.items():
            self.assertEqual(stats['count'],8);self.assertTrue(stats['input_lhs_strata'])
        for c in courts:
            p=c.provenance['parameters']
            self.assertNotIn('inspire_plan',c.provenance['case_ids'])
            self.assertAlmostEqual(c.provenance['expected_visitors'],sum(c.provenance['capacity_per_vendor'])*p['staff_load'])
            self.assertLessEqual(p['review_days'],min(p['protein_life_days'],p['vegetable_life_days']))
            self.assertLessEqual(p['stock_cover_days'],min(p['protein_life_days'],p['vegetable_life_days']))
            self.assertEqual(visitor_mean(c,dict(demand_mode='capacity',demand_scale=.1)),c.provenance['expected_visitors']*.1)
            if c.provenance['role']=='historical_joint_anchor' and c.provenance['case_ids']==['sora_t1']:
                self.assertEqual(c.duration,780)
            if c.provenance['venue_type'] in ('highway','leisure'):
                self.assertIn('assumption',c.provenance['structural_roles']['vendors'])

    def test_fefo_ledger_and_delivery_delay(self):
        c=make_court(Settings(),0,'test',10,8)
        c.provenance['parameters'].update(stock_cover_days=1.,lead_days=1,review_days=1,
            safety_days=0.,delivery_delay_probability=1.,protein_life_days=2,vegetable_life_days=2,staple_life_days=3)
        c.initial_stock=[dict(staple=2.,protein=2.,vegetable=2.) for _ in c.servers]
        inv=Inventory(c,7,'same')
        inv.begin_day(0);opening=inv.stock()
        used=[dict(staple=1.,protein=1.,vegetable=1.) for _ in c.servers]
        inv.finish_day(0,used,opening)
        self.assertTrue(inv.pending);self.assertTrue(all(o['due']==2 for o in inv.pending))
        # The policy sees announced lead time, not a future random delay outcome.
        self.assertAlmostEqual(inv.observations(0)[0][1],1/7)
        self.assertEqual(inv.begin_day(1),0.)
        waste=inv.begin_day(2);self.assertGreater(waste,0.)
        self.assertTrue(all(b['expires']>2 for rows in inv.batches for b in rows))
        audit=inv.audit()
        self.assertAlmostEqual(audit['initial_value']+audit['received_value'],audit['consumed_value']+audit['waste_value']+audit['on_hand_value'])

    def test_closed_calendar_and_parameter_application(self):
        c=make_court(Settings(),0,'test',10,8)
        rows,_=period(c,Settings(),__import__('kiosk_adaptive.design',fromlist=['Objective']).Objective('o',.4,.3,.3,.5,.5),
                      8,'calendar',10)
        self.assertTrue(any(not r['opened'] for r in rows))
        for r in rows:
            if not r['opened']:self.assertEqual(r['visitors'],0)
            self.assertAlmostEqual(r['behaviour']['taste_noise'],c.provenance['parameters']['taste_noise'])
            self.assertAlmostEqual(r['behaviour']['demand_cv'],c.provenance['parameters']['demand_cv'])


if __name__=='__main__':unittest.main()


class NamedTasteTests(unittest.TestCase):
    def test_named_tastes_follow_anchor_profiles_and_leave_venue_unchanged(self):
        import json
        from kiosk_adaptive.venues import data_file, TASTE_AXES
        anchors={a['name']:a for a in json.loads(data_file('menu_anchors.json').read_text(encoding='utf-8'))}
        latent=make_court(Settings(),3,'train',10,8)
        named=make_court(Settings(taste_model='named').validate(),3,'train',10,8)
        self.assertEqual(len(latent.dishes),len(named.dishes))
        for a,b in zip(latent.dishes,named.dishes):
            self.assertEqual((a.name,a.price,a.cook_minutes,a.counter),(b.name,b.price,b.cook_minutes,b.counter))
            base=anchors[b.name[3:].rsplit(' ',1)[0]]['taste']
            for k,v in zip(TASTE_AXES,b.taste):
                self.assertLessEqual(abs(v-base[k]),.3)
        self.assertNotEqual([d.taste for d in latent.dishes],[d.taste for d in named.dishes])
        with self.assertRaises(ValueError):Settings(taste_model='other').validate()
