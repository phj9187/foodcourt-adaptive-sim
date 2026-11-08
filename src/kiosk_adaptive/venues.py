"""Evidence-paired venue structures and stratified engineering sensitivity designs.

The supplied research is a historical purposive catalogue, not population data.
All menus/operations remain synthetic; every structural field carries its role.
"""
from dataclasses import asdict, replace
from functools import lru_cache
from itertools import combinations
import json
import math
import numpy as np
from .models import Court, Dish, CATEGORIES, data_file, rng_for

TYPES=('campus','mall','highway','airport','leisure')
HOURS=dict(campus=(6,11),mall=(10,12),highway=(16,24),airport=(12,24),leisure=(7,12))
UNSUPPORTED=()


def evidence():
    return json.loads(data_file('venue_evidence.json').read_text(encoding='utf-8'))


def maximin_lhs(n, dimensions, seed, candidates=24):
    rng=np.random.default_rng(seed)
    best, best_distance=None,-1.
    for _ in range(candidates):
        x=np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in range(dimensions)])
        if n==1:
            return x
        distance=((x[:,None,:]-x[None,:,:])**2).sum(-1)
        np.fill_diagonal(distance,np.inf)
        score=float(distance.min())
        if score>best_distance:
            best,best_distance=x,score
    return best


@lru_cache(maxsize=64)
def design_plan(seed, split, size, types, candidates):
    if not types or len(set(types))!=len(types) or not set(types)<=set(TYPES) or size<1:
        raise ValueError('valid nonempty venue types and positive design size required')
    ev=evidence();ranges={k:v for k,v in ev['assumed_ranges'].items() if k not in UNSUPPORTED};names=list(ranges)
    plan=[]
    for type_index,kind in enumerate(types):
        slots=list(range(type_index,size,len(types)))
        if not slots:
            continue
        rng=rng_for(seed,'venue_design',split,kind)
        cube=maximin_lhs(len(slots),len(names)+3,int(rng.integers(0,2**31)),candidates)
        cases=sorted([c for c in ev['cases'] if c['type']==kind and not c.get('planned')],key=lambda c:c['seats'])
        for local,(slot,point) in enumerate(zip(slots,cube)):
            p={k:float(lo+(hi-lo)*point[j]) for j,(k,(lo,hi)) in enumerate(ranges.items())}
            # First records retain a historical joint structure. Additional
            # records interpolate a single same-type pair with a shared t.
            if local<len(cases):
                a=b=cases[local];t=0.;role='historical_joint_anchor'
            else:
                a,b=cases[0],cases[-1]
                additional=len(slots)-len(cases)
                t=(local-len(cases)+point[-3])/max(1,additional)
                role='within_type_interpolation'
            roles={}
            def structure(key, fallback):
                if key in a and key in b:
                    roles[key]='historical_record' if a['id']==b['id'] else 'within_type_interpolation'
                    return (1-t)*a[key]+t*b[key]
                roles[key]='engineering_assumption_missing_field'
                return fallback
            vendors=max(1,int(round(structure('vendors',3+5*point[-2]))))
            seats=int(round(structure('seats',300)))
            hours=float(structure('hours',HOURS[kind][0]+point[-1]*(HOURS[kind][1]-HOURS[kind][0])))
            p['review_days']=min(max(1,int(round(p['review_days']))),
                                 max(1,int(p['protein_life_days'])),max(1,int(p['vegetable_life_days'])))
            p['lead_days']=max(1,int(round(p['lead_days'])))
            for key in ('protein_life_days','staple_life_days','vegetable_life_days'):
                # A delivered batch must survive at least one service day after
                # arrival; shorter lives make every order expire unsold.
                p[key]=max(1,int(round(p[key])),p['lead_days']+1)
            p['stock_cover_days']=min(p['stock_cover_days'],p['protein_life_days'],p['vegetable_life_days'])
            p['attention_limit']=max(1,int(round(p['attention_limit'])))
            opening=int(structure('opening_minute',480 if kind=='campus' else 600 if kind in ('mall','leisure') else 0))
            weekdays=a.get('weekdays',list(range(5)) if kind=='campus' else list(range(7)))
            if 'weekdays' not in a:roles['weekdays']='engineering_schedule_assumption'
            else:roles['weekdays']='historical_record'
            plan.append(dict(slot=slot,type=kind,coordinates=point.tolist(),parameters=p,
                vendors=vendors,seats=seats,hours=hours,opening_minute=opening,
                weekdays=weekdays,case_ids=list(dict.fromkeys([a['id'],b['id']])),
                source_ids=list(dict.fromkeys([a['source'],b['source']])),role=role,
                structural_roles=roles,interpolation=t,
                known_kiosks=int(round(structure('kiosks',0))) or None))
    return tuple(sorted(plan,key=lambda p:p['slot']))


def generate_venue(settings,index,split,visitors,design_size=None):
    size=design_size or settings.venue_samples
    batch,index_in_batch=divmod(index,size)
    plan=design_plan(settings.seed,f'{split}_batch{batch}',size,
                     tuple(settings.venue_types),settings.venue_lhs_candidates)
    spec=json.loads(json.dumps(plan[index_in_batch]));p=spec['parameters']
    rng=rng_for(settings.seed,'venue_menus',split,index)
    anchors=[a.copy() for a in json.loads(data_file('menu_anchors.json').read_text(encoding='utf-8')) if a['role']=='main']
    corrections={'Y15':'한식','Y16':'한식','A21':'한식','Y13':'동남아식'}
    for a in anchors:
        a['category']=corrections.get(a['id'],a['category'])
    pool_categories=list(dict.fromkeys(a['category'] for a in anchors))
    vendor_categories=list(rng.permutation(pool_categories))
    vendor_categories+=list(rng.choice(pool_categories,max(0,spec['vendors']-len(vendor_categories))))
    dishes=[];servers=[]
    for counter in range(spec['vendors']):
        category=vendor_categories[counter];pool=[a for a in anchors if a['category']==category]
        menu_count=int(np.clip(round(p['menus_per_vendor']*rng.uniform(1-.6*p['menu_imbalance'],1+.6*p['menu_imbalance'])),3,32))
        servers.append(int(rng.integers(1,max(2,math.ceil(p['workers_per_vendor'])+1))))
        center=rng.uniform(.2,.8,3)
        for slot in range(menu_count):
            anchor=pool[slot%len(pool)]
            price=max(500.,float(round((anchor['price_won'] or 5000)*p['price_multiplier']/500)*500))
            recipe=dict(staple=1,protein=int(rng.integers(1,3)),vegetable=int(rng.integers(1,3)))
            dishes.append(Dish(f'S{counter}M{slot:03d}','가상 '+anchor['name']+f' {counter+1}-{slot+1}',
                counter,category,price,recipe,tuple(np.clip(center+rng.normal(0,.12,3),0,1)),
                float(p['cook_minutes']*rng.uniform(.7,1.3)),float(rng.lognormal(0,.4)),False,
                ('wheat',) if category in ('분식','중식','일식','양식') else (),
                anchor['source_id'],False,'synthetic_multiplier_historical_anchor'))
    average_price=float(np.mean([d.price for d in dishes]))
    recipe_size=float(np.mean([sum(d.recipe.values()) for d in dishes]))
    unit=average_price*p['cost_fraction']/recipe_size
    costs=dict(staple=.6*unit,protein=1.5*unit,vegetable=.9*unit)
    duration=spec['hours']*60
    # Per-vendor capacities determine the demand mixture, avoiding a small
    # high-capacity vendor mix silently overloading one equally weighted counter.
    capacity=[servers[c]*duration/np.mean([d.cook_minutes for d in dishes if d.counter==c]) for c in range(len(servers))]
    potential=max(1.,sum(capacity)*p['staff_load'])
    portions=[v*p['staff_load']*settings.expected_purchase_rate*p['stock_cover_days'] for v in capacity]
    stock=[]
    for c,q in enumerate(portions):
        pool=[d for d in dishes if d.counter==c]
        stock.append({k:float(math.ceil(q*np.mean([d.recipe[k] for d in pool]))) for k in costs})
    kiosks=spec['known_kiosks'] or max(1,math.ceil(potential*p['order_minutes']/(duration*p['kiosk_load'])))
    ev=evidence()
    provenance=dict(type='source-informed synthetic venue',venue_type=spec['type'],
        source_ids=spec['source_ids'],case_ids=spec['case_ids'],role=spec['role'],
        structural_roles=spec['structural_roles'],interpolation=spec['interpolation'],
        coordinates=spec['coordinates'],coordinate_names=[k for k in ev['assumed_ranges'] if k not in UNSUPPORTED]+['structure_mix','missing_vendors','missing_hours'],
        parameters=p,unsupported_parameters=list(UNSUPPORTED),opening_minute=spec['opening_minute'],
        weekdays=spec['weekdays'],expected_visitors=potential,
        exogenous_demand_factor=float(rng_for(settings.seed,'external_population',split,index).lognormal(-.5*.5**2,.5)),
        exogenous_demand_basis='independent synthetic population multiplier; not fitted traffic',
        customer_budget_bounds=[.8*float(np.median([d.price for d in dishes])),1.6*float(np.median([d.price for d in dishes]))],
        customer_budget_basis='synthetic population frozen at venue creation; not recomputed after menu/price interventions',
        capacity_per_vendor=list(map(float,capacity)),demand_basis='synthetic_capacity_times_staff_load',
        source_checked_at=ev['checked_at'],range_status=ev['range_status'],
        caveats=['historical cases are not current measurements or representative population samples',
                 'same evidence anchors can appear across synthetic train/validation/test venues; not source-held-out',
                 'all prices are synthetic KRW; no foreign exchange conversion',
                 'all menu taste, recipes, costs, cooking times and allergens are assumptions',
                 'meal menu scaffold omits drinks/desserts/buffet and coordinated group seating',
                 'service queues drain at daily close; true overnight queue carry is not implemented'])
    return Court(f'{split}_{index:04d}',dishes,servers,kiosks,spec['seats'],duration,stock,costs,provenance)


def coverage(courts):
    report={}
    for kind in TYPES:
        selected=[c for c in courts if c.provenance.get('venue_type')==kind]
        if not selected:continue
        cube=np.asarray([c.provenance['coordinates'] for c in selected]);n=len(selected)
        strata=all(sorted((col*n).astype(int))==list(range(n)) for col in cube.T)
        def bounds(values):return [float(min(values)),float(max(values))]
        empty=[]
        for i,j in combinations(range(cube.shape[1]),2):
            cells={(min(3,int(4*x[i])),min(3,int(4*x[j]))) for x in cube}
            empty.append(1-len(cells)/16)
        actual=np.asarray([[len(c.servers),c.seats,c.duration/60,c.kiosks,sum(c.servers),
                           len(c.dishes),c.provenance['expected_visitors']] for c in selected],dtype=float)
        scaled=(actual-actual.min(0))/np.maximum(1e-9,np.ptp(actual,axis=0))
        distance=((scaled[:,None,:]-scaled[None,:,:])**2).sum(-1)
        np.fill_diagonal(distance,np.inf)
        histograms={name:dict(counts=np.histogram(actual[:,j],bins=5)[0].tolist(),
                             edges=np.histogram(actual[:,j],bins=5)[1].tolist(),
                             unique=len(set(actual[:,j].tolist())))
                    for j,name in enumerate(('vendors','seats','hours','kiosks','cooks','menus','visitors'))}
        report[kind]=dict(count=n,input_lhs_strata=strata,
            input_pairwise_4x4_empty_fraction=float(np.mean(empty)),
            vendor_counts=sorted({len(c.servers) for c in selected}),
            seats=bounds([c.seats for c in selected]),hours=bounds([c.duration/60 for c in selected]),
            menu_counts=bounds([len(c.dishes) for c in selected]),
            visitors=bounds([c.provenance['expected_visitors'] for c in selected]),
            cases=sorted({v for c in selected for v in c.provenance['case_ids']}),
            actual_min_distance=float(np.sqrt(distance.min())) if n>1 else None,
            output_histograms=histograms)
    return dict(types=report,warning='Input LHS is not a uniform output or joint population distribution. Missing cells are a diagnostic, not proof of impossible/possible coverage.')


def main():
    import argparse
    from pathlib import Path
    from .models import Settings
    parser=argparse.ArgumentParser()
    parser.add_argument('--types',nargs='+',choices=TYPES,default=['campus'])
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261105)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.count<len(args.types):parser.error('count must cover every selected type')
    settings=Settings(seed=args.seed,venue_types=tuple(args.types))
    courts=[generate_venue(settings,i,'design',0,args.count) for i in range(args.count)]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'court_catalog.json').write_text(json.dumps([asdict(c) for c in courts],ensure_ascii=False,indent=2),encoding='utf-8')
    report=coverage(courts)
    (output/'coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report['types'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
