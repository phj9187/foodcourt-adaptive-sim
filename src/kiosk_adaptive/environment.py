"""Chronological meal-period DES: shared kiosks, counter kitchens, shared seating.
No learner is updated with future completion or taste while processing arrivals.
"""
from collections import deque
import heapq,math
import numpy as np
from .models import rng_for,seed_for,CATEGORIES,POLICIES
from .customers import CustomerMemory,generate_customers,choose_menu


def simulate(court,settings,policy='none',theta=None,day=0,namespace='evaluation',visitors=300,attention='bounded',raw=False,controller=None,period_day=0,period_days=1,memory=None):
 if policy not in POLICIES:raise ValueError('unknown policy')
 if attention not in ('bounded','full'):raise ValueError('unknown attention mode')
 s=settings.validate();menus=court.dishes;m=len(menus)
 if memory is None:memory=CustomerMemory()
 day_seed=seed_for(s.seed,court.id,namespace,day)
 demand=rng_for(day_seed,'demand').lognormal(-.5*math.log1p(s.demand_cv**2),math.sqrt(math.log1p(s.demand_cv**2)))
 n=int(rng_for(day_seed,'count').poisson(visitors*demand))
 customers=generate_customers(s,court,day,namespace,n,visitors,memory)
 tastes=np.asarray([c['tastes'] for c in customers]).reshape(n,3)
 budgets=np.array([c['budget'] for c in customers])
 ps=np.array([c['ps'] for c in customers]);ws=np.array([c['ws'] for c in customers])
 allergy=np.array([c['allergy'] for c in customers],dtype=bool)
 dislikes=np.array([c['dislikes'] for c in customers],dtype=bool)
 disliked_category=np.array([c['excluded'] for c in customers],dtype=int)
 categories=CATEGORIES
 # Explicit ordinal kiosk inputs; hidden taste/noise remains private to the environment.
 taste_input=np.asarray([c['taste_input'] for c in customers]).reshape(n,3)
 price_input=np.array([c['price_input'] for c in customers]);wait_input=np.array([c['wait_input'] for c in customers])
 patience=np.array([c['patience'] for c in customers]);class_gap=np.array([c['class_gap'] for c in customers])
 # Type-specific synthetic peak/background mixture, not observed schedules.
 opening=court.provenance.get('opening_minute',480)
 if s.venue_type=='campus':clock_peaks=[690.,750.,1050.]
 elif s.venue_type in ('mall','leisure'):clock_peaks=[750.,1110.]
 else:clock_peaks=[opening+court.duration*v for v in (.2,.45,.7,.9)]
 locations=[x-opening for x in clock_peaks if 0<=x-opening<court.duration]
 if not locations:locations=[court.duration*.5]
 peaks=rng_for(day_seed,'peak_locations').choice(locations,n)
 concentrated=rng_for(day_seed,'peak').random(n)<s.peak_fraction
 arrivals=np.where(concentrated,peaks+rng_for(day_seed,'arrival_jitter').normal(0,court.duration*.06,n),
                   rng_for(day_seed,'arrival_background').uniform(0,court.duration,n))
 arrivals=np.clip(arrivals,0,np.nextafter(court.duration,0))
 fit=np.clip(1.-np.abs(tastes[:,None,:]-np.array([x.taste for x in menus])[None,:,:]).mean(axis=2),0,1)
 estimated=np.clip(fit+np.column_stack([rng_for(day_seed,'taste_noise',d.id).normal(0,s.taste_noise,n) for d in menus]),0,1) if n else fit.copy()
 shocks=np.column_stack([rng_for(day_seed,'choice',d.id).gumbel(0,s.choice_noise,n) for d in menus]) if n else fit.copy()
 # Display order/popularity affects exposure, not true taste or reward. No
 # automatic menu-count penalty or assumed choice-overload benefit is applied.
 priority=np.column_stack([rng_for(day_seed,'attention',d.id).random(n)-s.display_strength*(d.popularity/(1+d.popularity)+1/(1+j)) for j,d in enumerate(menus)]) if n else fit.copy()
 cook_factors=np.column_stack([rng_for(day_seed,'cook',d.id).lognormal(-.5*math.log1p(s.cook_cv**2),math.sqrt(math.log1p(s.cook_cv**2)),n) for d in menus]) if n else fit.copy()
 accept=rng_for(day_seed,'accept').random(n);outside=rng_for(day_seed,'outside').gumbel(0,s.choice_noise,n)
 cancel_draw=rng_for(day_seed,'cancel').random(n)
 stock=[dict(x) for x in court.initial_stock];consumed=[{k:0 for k in x} for x in stock]
 events=[];seq=0;trace=[]
 records=[dict(person=i,arrival=float(arrivals[i]),status='not_arrived',served=False,accepted=False,inspected=0,decision=0.,kiosk_wait=0.,cook_wait=0.,cook_minutes=0.,seat_wait=0.,total_wait=0.,taste=0.,affordability=0.,wait_utility=0.,revenue=0.,food_cost=0.,counter=None,menu=None) for i in range(n)]
 for i,r in enumerate(records):
  r['allergy_wheat']=bool(allergy[i])
  r['excluded_category']=categories[disliked_category[i]] if dislikes[i] else None
  r.update(customer_id=customers[i]['id'],repeat=customers[i]['repeat'],potential_visit=True,visited=customers[i]['visits'],budget=float(budgets[i]),behaviour_world=customers[i]['choice_world'])
 kiosk_queue=deque();free_kiosks=list(range(court.kiosks));heapq.heapify(free_kiosks);kiosk_busy=0.
 kitchen_queues=[deque() for _ in court.servers];busy=[[False]*c for c in court.servers];expected_end=[[0.]*c for c in court.servers]
 seat_queue=deque();free_seats=list(range(court.seats));heapq.heapify(free_seats)
 sales=np.zeros(m)
 prices=np.array([d.price for d in menus]);costs=np.array([sum(court.ingredient_costs[k]*v for k,v in d.recipe.items()) for d in menus])
 def schedule(time,kind,i,payload=None):
  nonlocal seq
  seq+=1;heapq.heappush(events,(float(time),seq,kind,int(i),payload))
 def log(time,kind,i,**kw):
  if raw:trace.append(dict(time=float(time),event=kind,person=int(i),**kw))
 def available(i,t):
  return [j for j,d in enumerate(menus) if t<court.duration and d.price<=budgets[i] and not(allergy[i] and 'wheat' in d.allergens) and not(dislikes[i] and d.category==categories[disliked_category[i]]) and all(stock[d.counter][k]>=v for k,v in d.recipe.items())]
 def wait_for(j,t):
  d=menus[j];c=d.counter
  queued=sum(menus[records[i]['chosen']].cook_minutes for i in kitchen_queues[c] if records[i]['status']=='waiting_kitchen')
  return max(0.,min(expected_end[c])-t)+queued/len(busy[c])+d.cook_minutes
 def decide(i,t,kiosk):
  r=records[i];r['status']='browsing';r['kiosk_wait']=t-r['arrival'];r['decision_start']=t;r['kiosk']=kiosk
  options=available(i,t)
  if not options:
   r['status']='no_feasible_menu';heapq.heappush(free_kiosks,kiosk);log(t,'no_feasible_menu',i);return
  considered=sorted(options,key=lambda j:priority[i,j])
  if attention=='bounded':considered=considered[:s.attention_limit]
  eta={j:wait_for(j,t) for j in options}
  perceived_seat_wait=len(seat_queue)*s.dining_minutes/max(1,court.seats) if court.seats else 0.
  perceived={j:eta[j]+perceived_seat_wait for j in options}
  utility={j:customers[i]['taste_weight']*estimated[i,j]-ps[i]*prices[j]/budgets[i]-ws[i]*perceived[j]/s.wait_scale+s.habit_strength*float(menus[j].category==customers[i]['habit']) for j in options}
  suggestions=[]
  if policy=='neural':
   if controller is None:raise ValueError('neural policy requires controller')
   # Permutation-equivariant candidates, no menu-ID embedding or future outcomes.
   congestion=sum(sum(v) for v in busy)/sum(court.servers)
   user=[*taste_input[i],budgets[i]/11000.,price_input[i],wait_input[i],max(0.,class_gap[i]-(t-r['arrival']))/60.]
   candidates=[]
   for j in options:
    d=menus[j];c=d.counter
    fraction=min(stock[c][k]/max(1,court.initial_stock[c][k]) for k in d.recipe)
    candidates.append([*user,*d.taste,d.price/11000.,d.cook_minutes/10.,
     (d.price-costs[j])/s.profit_scale,eta[j]/s.wait_scale,fraction,
     float(len(kitchen_queues[c]))/20.,court.servers[c]/3.,
     *[float(d.category==v) for v in categories],
     t/court.duration,congestion,len(menus)/60.,len({d.counter for d in menus})/6.,
     len(kiosk_queue)/20.,len(seat_queue)/100.,
     min(stock[c][k]/v for k,v in d.recipe.items())/300.,
     min(court.initial_stock[c][k]/v for k,v in d.recipe.items())/300.,
     period_day/period_days,(period_days-period_day)/period_days,
     court.duration/1440.,court.provenance.get('opening_minute',480)/1440.,court.seats/1000.,court.kiosks/20.,
     *[float(s.venue_type==kind) for kind in ('campus','mall','highway','airport','leisure')],
     *court.provenance.get('supply_observations',[[0.,0.,0.] for _ in court.servers])[c],
     court.provenance.get('weekday',0)/6.,court.provenance.get('days_to_next_service',1)/7.,
     d.popularity/(1+d.popularity),1/(1+j)])
   selected=controller(np.asarray(candidates,dtype=np.float32),rng_for(day_seed,'policy',i))
   if type(selected) is not int or not 0<=selected<len(options):raise ValueError('invalid neural action')
   suggestions=[options[selected]]
  elif policy=='random':suggestions=[int(rng_for(day_seed,'random_card',i).choice(options))]
  elif policy=='popular':suggestions=[max(options,key=lambda j:(menus[j].popularity,menus[j].id))]
  # Same card and same exposure mechanism for all recommending arms. A card
  # can enter consideration, but is never guaranteed acceptance or utility.
  exposed=bool(suggestions and accept[i]<s.openness)
  order=list(considered)
  if exposed:order=[suggestions[0]]+[j for j in order if j!=suggestions[0]]
  world=customers[i]['choice_world']
  threshold=(-1. if ws[i]>ps[i] and ws[i]>.7 else 0. if ps[i]>.7 else .5) if world=='lexicographic' else .5
  chosen,inspected=choose_menu(order,utility,shocks[i],prices,perceived,estimated[i],world,threshold)
  accepted=bool(exposed and chosen==suggestions[0])
  browsing=s.decision_base+s.per_item_minutes*len(inspected)+(s.recommendation_overhead if suggestions else 0.)
  reason='order'
  if s.outside_utility+outside[i]>utility[chosen]+shocks[i,chosen]:reason='outside_option'
  elif t+browsing+perceived[chosen]>r['arrival']+class_gap[i]:reason='predicted_wait_decline'
  duration=browsing+(s.payment_minutes if reason=='order' else 0.)
  r.update(chosen=chosen,accepted=accepted,recommendation_exposed=exposed,inspected=len(inspected),decision=duration,reason=reason,considered=[menus[j].id for j in inspected],suggestions=[menus[j].id for j in suggestions])
  schedule(t+duration,'decision_end',i,kiosk)
  log(t,'decision_start',i,kiosk=kiosk,chosen=menus[chosen].id,reason=reason)
 def dispatch_kiosks(t):
  while free_kiosks and kiosk_queue:
   i=kiosk_queue.popleft()
   if records[i]['status']!='waiting_kiosk':continue
   if t>=court.duration:records[i]['status']='closed';continue
   kiosk=heapq.heappop(free_kiosks);decide(i,t,kiosk)
 def dispatch_kitchen(c,t):
  for worker in range(len(busy[c])):
   if busy[c][worker]:continue
   while kitchen_queues[c] and records[kitchen_queues[c][0]]['status']!='waiting_kitchen':kitchen_queues[c].popleft()
   if not kitchen_queues[c]:continue
   i=kitchen_queues[c].popleft();r=records[i];j=r['chosen'];d=menus[j]
   busy[c][worker]=True;expected_end[c][worker]=t+d.cook_minutes
   duration=d.cook_minutes*cook_factors[i,j]
   r.update(status='cooking',cook_start=t,cook_wait=t-r['order_time'],cook_minutes=float(duration),worker=worker)
   schedule(t+duration,'food_ready',i,(c,worker));log(t,'cook_start',i,counter=c,worker=worker)
 def seated(i,t,seat=None):
  r=records[i];j=r['chosen'];r.update(status='served',served=True,seat_time=t,seat_wait=t-r['food_ready'],total_wait=t-r['arrival'],taste=float(fit[i,j]),affordability=float(max(0.,1.-prices[j]/budgets[i])),wait_utility=s.wait_scale/(s.wait_scale+t-r['arrival']))
  r['missed_class']=bool(t>r['arrival']+class_gap[i]);log(t,'served',i,counter=r['counter'])
  if seat is not None:schedule(t+float(rng_for(day_seed,'dining',i).uniform(.8*s.dining_minutes,1.2*s.dining_minutes)),'seat_release',i,seat)
 def dispatch_seats(t):
  while free_seats and seat_queue:
   i=seat_queue.popleft()
   if records[i]['status']=='waiting_seat':seated(i,t,heapq.heappop(free_seats))
 for i,arrival in enumerate(arrivals):schedule(arrival,'arrival',i)
 last_time=-1.
 while events:
  t,_,kind,i,payload=heapq.heappop(events)
  if t<last_time:raise AssertionError('time reversal')
  last_time=t;r=records[i]
  if kind=='arrival':
   if not customers[i]['visits']:r['status']='visit_declined';continue
   predicted_queue=(sum(records[v]['status']=='waiting_kiosk' for v in kiosk_queue)/court.kiosks)*s.decision_base
   if predicted_queue>min(patience[i],class_gap[i]):r['status']='queue_balk';log(t,'queue_balk',i);continue
   r['status']='waiting_kiosk';kiosk_queue.append(i);schedule(t+patience[i],'patience',i);dispatch_kiosks(t)
  elif kind=='patience':
   if r['status']=='waiting_kiosk':r['status']='queue_abandonment';log(t,'queue_abandonment',i)
  elif kind=='decision_end':
   kiosk_busy+=r['decision'];heapq.heappush(free_kiosks,payload);log(t,'decision_end',i,kiosk=payload)
   if r['reason']!='order':r['status']=r['reason']
   elif t>=court.duration:r['status']='closed_after_decision'
   elif r['chosen'] not in available(i,t):r['status']='sold_out_during_choice'
   else:
    j=r['chosen'];d=menus[j]
    for k,v in d.recipe.items():stock[d.counter][k]-=v;consumed[d.counter][k]+=v
    sales[j]+=1;r.update(status='waiting_kitchen',order_time=t,counter=d.counter,menu=d.id,revenue=float(prices[j]),food_cost=float(costs[j]));kitchen_queues[d.counter].append(i);dispatch_kitchen(d.counter,t)
    if cancel_draw[i]<s.postpayment_cancel_probability:
     schedule(min(t+patience[i],r['arrival']+class_gap[i]),'paid_patience',i)
   dispatch_kiosks(t)
  elif kind=='paid_patience':
   if r['status'] in ('waiting_kitchen','cooking','waiting_seat'):
    previous=r['status'];j=r['chosen'];d=menus[j]
    r.update(status='cancelled_'+previous,refund=r['revenue'],revenue=0.,cancellation_time=t)
    sales[j]-=1
    if previous=='waiting_kitchen':
     for k,v in d.recipe.items():stock[d.counter][k]+=v;consumed[d.counter][k]-=v
     r['food_cost']=0.
    else:r['prepared_food_loss']=r['food_cost']
    log(t,'paid_cancellation',i,stage=previous)
  elif kind=='food_ready':
   c,worker=payload;busy[c][worker]=False;expected_end[c][worker]=0.;dispatch_kitchen(c,t)
   if r['status']=='cancelled_cooking':log(t,'discard_prepared',i,counter=c);continue
   r.update(status='waiting_seat',food_ready=t);log(t,'food_ready',i,counter=c,worker=worker)
   if court.seats:seat_queue.append(i);dispatch_seats(t)
   else:seated(i,t)
  elif kind=='seat_release':heapq.heappush(free_seats,payload);dispatch_seats(t)
 terminal={'served','no_feasible_menu','closed','closed_after_decision','outside_option','predicted_wait_decline','queue_abandonment','sold_out_during_choice','visit_declined','queue_balk','cancelled_waiting_kitchen','cancelled_cooking','cancelled_waiting_seat'}
 if any(r['status'] not in terminal for r in records):raise AssertionError('unfinished customer')
 for c in range(len(stock)):
  for k,v in stock[c].items():
   if v<0 or not math.isclose(court.initial_stock[c][k],v+consumed[c][k]):raise AssertionError('inventory conservation')
 served=[r for r in records if r['served']];den=max(1,n);count=len(served)
 avg=lambda key:sum(r[key] for r in served)/max(1,count)
 revenue=sum(r['revenue'] for r in records);food_cost=sum(r['food_cost'] for r in records)
 # Calendar expiry is accounted by Inventory in the multi-day driver.
 waste_cost=0.;profit=revenue-food_cost
 result=dict(visitors=n,served=count,served_fraction=count/den,taste_per_buyer=avg('taste'),mean_total_wait=avg('total_wait'),mean_kiosk_wait=avg('kiosk_wait'),mean_decision_time=avg('decision'),mean_cook_wait=avg('cook_wait'),mean_cook_minutes=avg('cook_minutes'),mean_seat_wait=avg('seat_wait'),mean_inspected=sum(r['inspected'] for r in records)/den,revenue=revenue,food_cost=food_cost,waste_cost=waste_cost,contribution_profit=profit,profit_per_visitor=profit/den,store=profit/(den*s.profit_scale),kiosk_busy_minutes=kiosk_busy,kiosk_utilization=kiosk_busy/(court.kiosks*court.duration),overtime_minutes=max(0.,max((r['food_ready'] for r in served),default=court.duration)-court.duration),accepted_recommendations=sum(r['accepted'] for r in records),missed_class=sum(r.get('missed_class',False) for r in served),active_menus=m,menu_count=m)
 for status in sorted(terminal):result['count_'+status]=sum(r['status']==status for r in records)
 bounds=court.provenance.get('customer_budget_bounds',[s.budget_low,s.budget_high])
 for name,mask in [('low_budget',budgets<sum(bounds)/2),('high_wait_sensitivity',ws>.7)]:
  group=[r for i,r in enumerate(records) if mask[i]]; size=len(group)
  result['visitors_'+name]=size
  result['service_'+name]=sum(r['served'] for r in group)/max(1,size)
 result['counter_revenue']=[sum(r['revenue'] for r in records if r['counter']==c) for c in range(len(court.servers))]
 result.update(potential_visitors=n,actual_visitors=sum(r['visited'] for r in records),
   repeat_opportunities=sum(r['repeat'] for r in records),
   refunds=sum(r.get('refund',0.) for r in records),
   prepared_food_loss=sum(r.get('prepared_food_loss',0.) for r in records),
   recommendation_exposures=sum(r.get('recommendation_exposed',False) for r in records),
   wait_p90=float(np.quantile([r['total_wait'] for r in served],.9)) if served else 0.,
   wait_p95=float(np.quantile([r['total_wait'] for r in served],.95)) if served else 0.)
 result['choice_world_counts']={w:sum(c['choice_world']==w for c in customers) for w in ('mixed_logit','satisficing','lexicographic')}
 for i,r in enumerate(records):
  if customers[i]['repeat'] and customers[i]['visits']:
   # Experience is a latent behavioural mechanism, never the sampled F score.
   weights=np.array([customers[i]['taste_weight'],ps[i],ws[i]])
   experience=float(weights@np.array([r['taste'],r['affordability'],r['wait_utility']])/weights.sum()) if r['served'] else 0.
   category=menus[r['chosen']].category if r['served'] else None
   memory.observe(customers[i]['id'],experience,category,s.memory_strength)
 return result,dict(records=records,trace=trace,initial_stock=court.initial_stock,consumed=consumed,leftover=stock) if raw else None
