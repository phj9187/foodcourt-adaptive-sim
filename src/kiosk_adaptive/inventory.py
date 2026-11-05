"""Daily FEFO stock and lagged procurement; not temperature/recipe safety data.

Expiry and deliveries are processed at service opening, in whole calendar days.
The within-service DES drains its queue before returning, so this is NOT a
continuous overnight kitchen model. Bookkeeping is checked for every vendor.
"""
import copy
import math
from .models import rng_for


class Inventory:
    def __init__(self,court,seed,namespace):
        self.court=court;self.seed=seed;self.namespace=namespace
        self.parameters=court.provenance['parameters']
        p=self.parameters
        self.lives=dict(staple=p['staple_life_days'],protein=p['protein_life_days'],vegetable=p['vegetable_life_days'])
        self.batches=[[dict(ingredient=k,quantity=float(q),expires=int(self.lives[k]))
                       for k,q in stock.items()] for stock in court.initial_stock]
        self.pending=[]
        self.forecast=[{k:q/p['stock_cover_days'] for k,q in stock.items()} for stock in court.initial_stock]
        self.initial=copy.deepcopy(court.initial_stock)
        self.received=[{k:0. for k in row} for row in self.initial]
        self.consumed=copy.deepcopy(self.received);self.wasted=copy.deepcopy(self.received)

    def stock(self):
        rows=[{k:0. for k in self.court.ingredient_costs} for _ in self.batches]
        for c,batches in enumerate(self.batches):
            for b in batches:rows[c][b['ingredient']]+=b['quantity']
        return rows

    def begin_day(self,day):
        waste=[{k:0. for k in self.court.ingredient_costs} for _ in self.batches]
        for c,batches in enumerate(self.batches):
            for b in batches:
                if b['expires']<=day:
                    waste[c][b['ingredient']]+=b['quantity'];self.wasted[c][b['ingredient']]+=b['quantity']
            self.batches[c]=[b for b in batches if b['expires']>day]
        arrivals=[o for o in self.pending if o['due']<=day]
        self.pending=[o for o in self.pending if o['due']>day]
        for o in arrivals:
            for k,q in o['quantity'].items():
                self.batches[o['counter']].append(dict(ingredient=k,quantity=q,expires=day+int(self.lives[k])))
                self.received[o['counter']][k]+=q
        return sum(q*self.court.ingredient_costs[k] for row in waste for k,q in row.items())

    def finish_day(self,day,used,opening_stock,opened=True):
        for c,row in enumerate(used):
            for k,quantity in row.items():
                remaining=quantity
                for b in sorted(self.batches[c],key=lambda b:b['expires']):
                    if b['ingredient']!=k:continue
                    amount=min(remaining,b['quantity']);b['quantity']-=amount;remaining-=amount
                if remaining>1e-7:raise AssertionError('FEFO stock overconsumed')
                self.consumed[c][k]+=quantity
                # Uses observed past consumption and an initial capacity-based
                # prior, never tomorrow's actual visitor draw.
                if opened:self.forecast[c][k]=.5*self.forecast[c][k]+.5*quantity
            self.batches[c]=[b for b in self.batches[c] if b['quantity']>1e-9]
        stock=self.stock();p=self.parameters
        if day%p['review_days']==0:
            for c,row in enumerate(stock):
                quantities={}
                for k,q in row.items():
                    cover=min(self.lives[k],p['lead_days']+p['review_days']+p['safety_days'])
                    pipeline=sum(o['quantity'][k] for o in self.pending if o['counter']==c)
                    quantities[k]=float(max(0,math.ceil(self.forecast[c][k]*cover-q-pipeline)))
                if any(quantities.values()):
                    delay=int(rng_for(self.seed,self.court.id,self.namespace,'delivery_delay',day,c).random()<p['delivery_delay_probability'])
                    self.pending.append(dict(counter=c,due=day+p['lead_days']+delay,
                                             expected_due=day+p['lead_days'],quantity=quantities))
        for c,row in enumerate(stock):
            for k,q in row.items():
                assert math.isclose(opening_stock[c][k]-used[c][k],q,abs_tol=1e-6)
                assert math.isclose(self.initial[c][k]+self.received[c][k],
                    self.consumed[c][k]+self.wasted[c][k]+q,abs_tol=1e-6)
        return stock

    def observations(self,day):
        output=[]
        for c,batches in enumerate(self.batches):
            remaining=[b['expires']-day for b in batches if b['quantity']>0]
            orders=[o for o in self.pending if o['counter']==c]
            output.append([min(remaining,default=0)/10.,
                min((max(0,o['expected_due']-day) for o in orders),default=0)/7.,
                sum(sum(o['quantity'].values()) for o in orders)/1000.])
        return output

    def audit(self):
        costs=self.court.ingredient_costs
        value=lambda rows:sum(q*costs[k] for row in rows for k,q in row.items())
        return dict(initial_value=value(self.initial),received_value=value(self.received),
                    consumed_value=value(self.consumed),waste_value=value(self.wasted),
                    on_hand_value=value(self.stock()),
                    pipeline_value=sum(value([o['quantity']]) for o in self.pending),
                    pending=copy.deepcopy(self.pending))
