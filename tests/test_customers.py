import copy
from dataclasses import replace
import unittest
import numpy as np
from kiosk_adaptive.customers import CustomerMemory,generate_customers,WORLDS
from kiosk_adaptive.design import make_court,Objective
from kiosk_adaptive.environment import simulate
from kiosk_adaptive.models import Settings,Court,Dish
from kiosk_adaptive.study import evaluate,visitor_mean
from kiosk_adaptive.policy import ActorCritic


class CustomerTests(unittest.TestCase):
    def test_joint_traits_mixed_population_and_fixed_intervention_budgets(self):
        s=Settings(repeat_share=1.)
        c=make_court(s,0,'test',50)
        a=generate_customers(s,c,0,'paired',2000,1000,CustomerMemory())
        self.assertEqual({p['choice_world'] for p in a},set(WORLDS))
        corr=np.corrcoef([p['budget'] for p in a],[p['ps'] for p in a])[0,1]
        self.assertLess(corr,-.5)
        changed=copy.deepcopy(c)
        changed.dishes=[replace(d,price=d.price*2) for d in changed.dishes]
        b=generate_customers(s,changed,0,'paired',2000,1000,CustomerMemory())
        self.assertEqual([p['budget'] for p in a],[p['budget'] for p in b])
        self.assertEqual(c.provenance['expected_visitors'],changed.provenance['expected_visitors'])
        config=dict(demand_mode='exogenous',visitors=100,demand_scale=1.)
        changed.servers=[v*3 for v in changed.servers]
        self.assertEqual(visitor_mean(c,config),visitor_mean(changed,config))

    def test_repeat_identity_and_policy_dependent_attendance(self):
        s=Settings(repeat_share=1.,memory_strength=.8)
        c=make_court(s,0,'test',50)
        good=CustomerMemory();bad=CustomerMemory()
        first=generate_customers(s,c,0,'same',400,100,good)
        for p in first:
            good.experiences[p['id']]=(1.,'한식')
            bad.experiences[p['id']]=(0.,'한식')
        a=generate_customers(s,c,1,'same',400,100,good)
        b=generate_customers(s,c,1,'same',400,100,bad)
        self.assertEqual([p['id'] for p in a],[p['id'] for p in b])
        self.assertGreater(sum(p['visits'] for p in a),sum(p['visits'] for p in b))
        profiles={p['id']:p for p in first}
        for p in a:
            self.assertEqual(p['budget'],profiles[p['id']]['budget'])
            self.assertEqual(p['ps'],profiles[p['id']]['ps'])

    def test_same_card_mechanism_for_identical_recommendations(self):
        s=Settings();c=make_court(s,0,'test',70)
        def popular(x,rng):return int(np.argmax(x[:,-2]))
        a,ra=simulate(c,s,'popular',visitors=70,raw=True)
        b,rb=simulate(c,s,'neural',visitors=70,raw=True,controller=popular)
        self.assertEqual(a,b)
        self.assertEqual(ra['records'],rb['records'])

    def test_postpayment_refunds_preserve_inventory_and_charge_prepared_food_once(self):
        s=Settings(seed=7,repeat_share=0,dominant_choice_share=1.,postpayment_cancel_probability=1.,
                   patience_minutes=.04,outside_utility=-100.,decision_base=0.,per_item_minutes=0.,payment_minutes=0.,
                   dining_minutes=100.,peak_fraction=1.,cook_cv=.01)
        d=Dish('S0M000','test',0,'한식',1000.,{'staple':1.},(.5,.5,.5),2.,1.,False,(),'test',False,'synthetic')
        c=Court('refund',[d],[1],100,1,10.,[{'staple':1000.}],{'staple':100.},{})
        row,raw=simulate(c,s,visitors=100,raw=True)
        self.assertGreater(row['count_cancelled_cooking'],0)
        self.assertGreater(row['count_cancelled_waiting_kitchen'],0)
        self.assertEqual(row['revenue'],sum(r['revenue'] for r in raw['records']))
        self.assertAlmostEqual(row['food_cost'],100.*raw['consumed'][0]['staple'])
        self.assertAlmostEqual(row['prepared_food_loss'],sum(r.get('prepared_food_loss',0.) for r in raw['records']))
        self.assertGreater(row['refunds'],0.)
        for r in raw['records']:
            if r['status']=='cancelled_waiting_kitchen':self.assertEqual(r['food_cost'],0.)
        c.dishes=[replace(d,cook_minutes=.001)]
        row,raw=simulate(c,replace(s,patience_minutes=.2),visitors=100,raw=True)
        self.assertGreater(row['count_cancelled_waiting_seat'],0)
        self.assertAlmostEqual(row['food_cost'],100.*raw['consumed'][0]['staple'])

    def test_controls_and_worlds_keep_within_objective_pairing(self):
        s=Settings()
        cfg=dict(evaluation_repeats=1,test_courts=1,period_days=2,visitors=10,
                 demand_mode='fixed',behaviour_worlds=list(WORLDS),comparison_controls=True)
        objective=Objective('test',.3,.3,.4,.6,.5)
        pairs=evaluate(ActorCritic(8),s,objective,cfg,'test')
        self.assertEqual(len(pairs),6)
        for p in pairs:
            self.assertAlmostEqual(p['delta'],p['ai_score']-p['baseline_score'])
            self.assertEqual(set(p['controls']),{'random','popular','rule_fit','rule_fast','rule_margin','rule_objective'})
            for a,b in zip(p['baseline'],p['ai']):
                self.assertEqual(a['potential_visitors'],b['potential_visitors'])
            for arm in p['controls'].values():
                self.assertAlmostEqual(arm['ai_minus_control'],p['ai_score']-arm['score'])


if __name__=='__main__':unittest.main()
