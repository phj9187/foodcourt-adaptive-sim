"""Daily FEFO stock and lagged procurement; not temperature/recipe safety data.

Expiry and deliveries are processed at service opening, in whole calendar days.
The within-service DES drains its queue before returning, so this is NOT a
continuous overnight kitchen model. Bookkeeping is checked for every vendor.
"""
import copy
import math
from .models import rng_for


def demand_stock(court,visitors,settings):
    """Opening stock sized to the exogenous visitor mean, not kitchen capacity.

    Counter shares follow synthetic capacity; quantities cover stock_cover_days
    of expected purchases. Future realized visitors are never used.
    """
    p=court.provenance['parameters'];capacity=court.provenance.get('capacity_per_vendor') or [1.]*len(court.servers)
    # Opening stock must bridge the lead time until the first possible delivery;
    # otherwise every venue starts with an artificial stockout.
    life=min(p['protein_life_days'],p['vegetable_life_days'],p['staple_life_days'])
    cover=max(p['stock_cover_days'],min(p['lead_days']+p['safety_days'],life))
    court.provenance['opening_cover_days']=cover
    total=max(1e-9,sum(capacity));rows=[]
    for c,share in enumerate(capacity):
        pool=[d for d in court.dishes if d.counter==c]
        portions=visitors*settings.expected_purchase_rate*cover*share/total
        rows.append({k:float(math.ceil(portions*(sum(d.recipe.get(k,0) for d in pool)/max(1,len(pool))))) for k in court.ingredient_costs})
    return rows


class Inventory:
    def __init__(self,court,seed,namespace):
        self.court=court;self.seed=seed;self.namespace=namespace
        self.parameters=court.provenance['parameters']
        p=self.parameters
        self.lives=dict(staple=p['staple_life_days'],protein=p['protein_life_days'],vegetable=p['vegetable_life_days'])
        self.batches=[[dict(ingredient=k,quantity=float(q),expires=int(self.lives[k]))
                       for k,q in stock.items()] for stock in court.initial_stock]
        self.pending=[]
        cover=court.provenance.get('opening_cover_days',p['stock_cover_days'])
        self.forecast=[{k:q/cover for k,q in stock.items()} for stock in court.initial_stock]
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

    def usable(self,c,k,day):
        """On-hand units that can still be sold before expiring, at the forecast rate (FEFO)."""
        rate=max(1e-9,self.forecast[c][k]);covered=0.;total=0.
        for b in sorted((b for b in self.batches[c] if b['ingredient']==k),key=lambda b:b['expires']):
            # Demand that can be met from this batch before its own expiry.
            room=max(0.,rate*(b['expires']-day-1)-covered)
            take=min(b['quantity'],room);covered+=take;total+=take
        return total

    def finish_day(self,day,used,opening_stock,opened=True,lost=None):
        for c,row in enumerate(used):
            for k,quantity in row.items():
                remaining=quantity
                for b in sorted(self.batches[c],key=lambda b:b['expires']):
                    if b['ingredient']!=k:continue
                    amount=min(remaining,b['quantity']);b['quantity']-=amount;remaining-=amount
                if remaining>1e-7:raise AssertionError('FEFO stock overconsumed')
                self.consumed[c][k]+=quantity
                # Uses observed past consumption plus censored (stock-blocked)
                # demand, never tomorrow's actual visitor draw. A stockout day
                # cannot lower the forecast, avoiding a self-reinforcing spiral.
                if opened:
                    unmet=float(lost[c].get(k,0.)) if lost else 0.
                    updated=.5*self.forecast[c][k]+.5*(quantity+unmet)
                    self.forecast[c][k]=max(self.forecast[c][k],updated) if unmet>0 else updated
            self.batches[c]=[b for b in self.batches[c] if b['quantity']>1e-9]
        stock=self.stock();p=self.parameters
        if day%p['review_days']==0:
            for c,row in enumerate(stock):
                quantities={}
                for k,q in row.items():
                    cover=min(self.lives[k],p['lead_days']+p['review_days']+p['safety_days'])
                    pipeline=sum(o['quantity'][k] for o in self.pending if o['counter']==c)
                    # Count only stock that can still be sold before it expires.
                    quantities[k]=float(max(0,math.ceil(self.forecast[c][k]*cover-self.usable(c,k,day)-pipeline)))
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
