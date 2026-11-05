import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from kiosk_adaptive.design import lhs_objectives, make_court, Objective
from kiosk_adaptive.models import Settings
from kiosk_adaptive.environment import simulate
from kiosk_adaptive.policy import ActorCritic, Collector, FEATURES, update_ppo
from kiosk_adaptive.study import period, run


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.s = Settings()
        self.o = Objective('test', .4,.3,.3,.5,.7)

    def test_lhs_strata_and_simplex(self):
        objectives, cube = lhs_objectives(12,4)
        for column in np.array(cube).T:
            self.assertEqual(sorted((column*12).astype(int)),list(range(12)))
        for o in objectives:
            self.assertAlmostEqual(o.taste+o.affordability+o.waiting,1.)
        self.assertEqual(lhs_objectives(12,4),(objectives,cube))

    def test_variable_courts_and_split_separation(self):
        courts = [make_court(self.s,i,'train',50) for i in range(20)]
        self.assertGreater(len({len(c.servers) for c in courts}),1)
        for c in courts:
            self.assertTrue(2<=len(c.servers)<=16)
            self.assertGreaterEqual(len({d.category for d in c.dishes}),2)
            for counter in range(len(c.servers)):
                self.assertTrue(3<=sum(d.counter==counter for d in c.dishes)<=32)
        self.assertNotEqual(courts[0].dishes,make_court(self.s,0,'test',50).dishes)

    def test_permutation_equivariance_and_variable_size(self):
        model = ActorCritic()
        x = torch.randn(9,FEATURES); perm = torch.randperm(9)
        p,v = model(x); q,w = model(x[perm])
        torch.testing.assert_close(q.probs,p.probs[perm]);torch.testing.assert_close(v,w)
        for n in (1,3,64):
            dist,_=model(torch.randn(n,FEATURES))
            self.assertAlmostEqual(float(dist.probs.detach().sum()),1.,places=6)

    def test_hard_masks_and_no_future_fields(self):
        c=make_court(self.s,0,'test',150)
        calls=[]
        def choose(x,rng):
            self.assertEqual(x.shape[1],FEATURES)
            self.assertTrue(np.isfinite(x).all()); calls.append(x.copy())
            return 0
        row,raw=simulate(c,self.s,'neural',visitors=150,raw=True,controller=choose)
        self.assertGreater(len(calls),0)
        self.assertEqual(row['visitors'],sum(v for k,v in row.items() if k.startswith('count_')))
        menu_map={d.id:d for d in c.dishes}
        for record in raw['records']:
            for menu_id in record.get('suggestions',[]):
                dish=menu_map[menu_id]
                self.assertFalse(record['allergy_wheat'] and 'wheat' in dish.allergens)
                self.assertNotEqual(dish.category,record['excluded_category'])
        # At all times masks must leave only stock-feasible candidates.
        for x in calls:
            self.assertTrue((x[:,14]>0).all())
        empty=copy.deepcopy(c)
        empty.initial_stock=[{k:0 for k in d} for d in c.initial_stock]
        def forbidden(*args):raise AssertionError('empty feasible set sent to actor')
        result,_=simulate(empty,self.s,'neural',visitors=20,controller=forbidden)
        self.assertEqual(result['served'],0)

    def test_paired_randomness_and_continuous_inventory(self):
        c=make_court(self.s,1,'test',60)
        c.provenance['weekdays']=list(range(7))
        a,_=period(c,self.s,self.o,4,'test',60,changes=True)
        b,samples=period(c,self.s,self.o,4,'test',60,ActorCritic(),changes=True)
        self.assertEqual([r['visitors'] for r in a],[r['visitors'] for r in b])
        for rows in (a,b):
            for day in rows:
                inv=day['inventory']
                for initial,used,left,carried in zip(inv['initial'],inv['consumed'],inv['leftover'],inv['carried']):
                    for k in initial:
                        self.assertAlmostEqual(initial[k],used[k]+left[k])
                        self.assertAlmostEqual(carried[k],left[k])
                ledger=inv['supply']
                self.assertAlmostEqual(ledger['initial_value']+ledger['received_value'],
                    ledger['consumed_value']+ledger['waste_value']+ledger['on_hand_value'])
        self.assertGreater(len(samples),0)

    def test_no_forced_improvement_and_update(self):
        s=Settings(openness=0,recommendation_overhead=1.)
        c=make_court(s,0,'test',60)
        c.provenance['weekdays']=list(range(7))
        baseline,_=period(c,s,self.o,3,'same',60)
        model=ActorCritic(); before=copy.deepcopy(model.state_dict())
        ai,samples=period(c,s,self.o,3,'same',60,model,learn=True)
        self.assertEqual(sum(r['accepted'] for r in ai),0)
        result=update_ppo(model,torch.optim.Adam(model.parameters(),lr=.001),samples)
        self.assertGreater(result['samples'],0)
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items()))

    def test_resume_matches_uninterrupted_and_config_guard(self):
        cfg=dict(lhs_samples=1,replicas=1,train_courts=2,validation_courts=1,test_courts=1,
                 training_blocks=2,period_days=2,visitors=8,evaluation_repeats=1,
                 validation_every=2,ppo_epochs=1,hidden=8,demand_mode='fixed')
        import kiosk_adaptive.study as study
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            expected=run(cfg,a)
            real=study.update_ppo
            counter=[0]
            def interrupt(*args,**kwargs):
                counter[0]+=1
                if counter[0]==2:raise RuntimeError('simulated interruption')
                return real(*args,**kwargs)
            with patch.object(study,'update_ppo',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):run(cfg,b)
            actual=run(cfg,b)
            self.assertEqual(expected,actual)
            for checkpoint in Path(a).rglob('checkpoint.pt'):
                other=Path(b)/checkpoint.relative_to(a)
                x=torch.load(checkpoint,weights_only=False);y=torch.load(other,weights_only=False)
                for key in x['model']:torch.testing.assert_close(x['model'][key],y['model'][key],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'configuration/source changed'):
                run(dict(cfg,visitors=9),b)


if __name__=='__main__':unittest.main()
