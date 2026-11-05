"""Chronological meal-period DES: shared kiosks, counter kitchens, shared seating.
No learner is updated with future completion or taste while processing arrivals.
"""
from collections import deque
import heapq,math
import numpy as np
from .models import rng_for,seed_for,softmax


def simulate(court,settings,policy='none',theta=None,day=0,namespace='evaluation',visitors=300,attention='bounded',raw=False,controller=None):
 if policy not in ('none','neural'):raise ValueError('unknown policy')
 if attention not in ('bounded','full'):raise ValueError('unknown attention mode')
 s=settings;menus=court.dishes;m=len(menus)
 day_seed=seed_for(s.seed,court.id,namespace,day)
 demand=rng_for(day_seed,'demand').lognormal(-.5*math.log1p(s.demand_cv**2),math.sqrt(math.log1p(s.demand_cv**2)))
 n=int(rng_for(day_seed,'count').poisson(visitors*demand))
 tastes=np.clip(rng_for(day_seed,'tastes').normal(.5,.23,(n,3)),0,1)
 budgets=rng_for(day_seed,'budgets').uniform(s.budget_low,s.budget_high,n)
 ps=rng_for(day_seed,'price_sensitivity').uniform(.3,1.1,n);ws=rng_for(day_seed,'wait_sensitivity').uniform(.3,1.1,n)
 allergy=rng_for(day_seed,'allergy').random(n)<.05
 dislikes=rng_for(day_seed,'dislikes').random(n)<.10
 disliked_category=rng_for(day_seed,'disliked_category').integers(0,4,n)
 categories=('분식','중식','일식','양식')
 # Explicit ordinal kiosk inputs; hidden taste/noise remains private to the environment.
 taste_input=np.round(tastes*4)/4
 price_input=np.round(np.clip((ps-.3)/.8,0,1)*4)/4
 wait_input=np.round(np.clip((ws-.3)/.8,0,1)*4)/4
 patience=rng_for(day_seed,'patience').exponential(s.patience_minutes,n)/np.maximum(.2,ws)
 class_gap=rng_for(day_seed,'class_gap').uniform(20.,60.,n)
 # Two synthetic class-release peaks. These are not measured campus footfall.
 peaks=np.where(rng_for(day_seed,'peak').random(n)<.65,.36,.62)*court.duration
 arrivals=np.clip(peaks+rng_for(day_seed,'arrival_jitter').normal(0,court.duration*.13,n),0,np.nextafter(court.duration,0))
 fit=np.clip(1.-np.abs(tastes[:,None,:]-np.array([x.taste for x in menus])[None,:,:]).mean(axis=2),0,1)
 estimated=np.clip(fit+np.column_stack([rng_for(day_seed,'taste_noise',d.id).normal(0,s.taste_noise,n) for d in menus]),0,1) if n else fit.copy()
 shocks=np.column_stack([rng_for(day_seed,'choice',d.id).gumbel(0,s.choice_noise,n) for d in menus]) if n else fit.copy()
 priority=np.column_stack([rng_for(day_seed,'attention',d.id).random(n) for d in menus]) if n else fit.copy()
 cook_factors=np.column_stack([rng_for(day_seed,'cook',d.id).lognormal(-.5*math.log1p(s.cook_cv**2),math.sqrt(math.log1p(s.cook_cv**2)),n) for d in menus]) if n else fit.copy()
 accept=rng_for(day_seed,'accept').random(n);outside=rng_for(day_seed,'outside').gumbel(0,s.choice_noise,n)
 stock=[dict(x) for x in court.initial_stock];consumed=[{k:0 for k in x} for x in stock]
 events=[];seq=0;trace=[]
 records=[dict(person=i,arrival=float(arrivals[i]),status='not_arrived',served=False,accepted=False,inspected=0,decision=0.,kiosk_wait=0.,cook_wait=0.,cook_minutes=0.,seat_wait=0.,total_wait=0.,taste=0.,affordability=0.,wait_utility=0.,revenue=0.,food_cost=0.,counter=None,menu=None) for i in range(n)]
 for i,r in enumerate(records):
  r['allergy_wheat']=bool(allergy[i])
  r['excluded_category']=categories[disliked_category[i]] if dislikes[i] else None
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
  queued=sum(menus[records[i]['chosen']].cook_minutes for i in kitchen_queues[c])
  return max(0.,min(expected_end[c])-t)+queued/len(busy[c])+d.cook_minutes
 def decide(i,t,kiosk):
  r=records[i];r['status']='browsing';r['kiosk_wait']=t-r['arrival'];r['decision_start']=t;r['kiosk']=kiosk
  options=available(i,t)
  if not options:
   r['status']='no_feasible_menu';heapq.heappush(free_kiosks,kiosk);log(t,'no_feasible_menu',i);return
  considered=sorted(options,key=lambda j:priority[i,j])
  if attention=='bounded':considered=considered[:s.attention_limit]
  eta={j:wait_for(j,t) for j in options}
  utility={j:2.5*estimated[i,j]-ps[i]*prices[j]/budgets[i]-ws[i]*eta[j]/s.wait_scale for j in options}
  own=max(considered,key=lambda j:utility[j]+shocks[i,j])
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
     len(kiosk_queue)/20.,len(seat_queue)/100.])
   selected=controller(np.asarray(candidates,dtype=np.float32),rng_for(day_seed,'policy',i))
   if type(selected) is not int or not 0<=selected<len(options):raise ValueError('invalid neural action')
   suggestions=[options[selected]]
  suggested=max(suggestions,key=lambda j:utility[j]+shocks[i,j]) if suggestions else None
  # Accept against a seen reference, never against an unseen optimum.
  reference=considered[0];gain=utility[suggested]-utility[reference] if suggested is not None else 0.
  probability=s.openness/(1.+math.exp(-max(-40.,min(40.,3*gain)))) if suggestions else 0.
  accepted=bool(accept[i]<probability);chosen=suggested if accepted else own
  inspected=set(considered[:1] if accepted else considered);inspected.update(suggestions)
  browsing=s.decision_base+s.per_item_minutes*len(inspected)+(s.recommendation_overhead if suggestions else 0.)
  reason='order'
  if s.outside_utility+outside[i]>utility[chosen]+shocks[i,chosen]:reason='outside_option'
  elif t+browsing+eta[chosen]>r['arrival']+class_gap[i]:reason='predicted_wait_decline'
  duration=browsing+(s.payment_minutes if reason=='order' else 0.)
  r.update(chosen=chosen,accepted=accepted,inspected=len(inspected),decision=duration,reason=reason,considered=[menus[j].id for j in considered],suggestions=[menus[j].id for j in suggestions])
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
   if busy[c][worker] or not kitchen_queues[c]:continue
   i=kitchen_queues[c].popleft();r=records[i];j=r['chosen'];d=menus[j]
   busy[c][worker]=True;expected_end[c][worker]=t+d.cook_minutes
   duration=d.cook_minutes*cook_factors[i,j]
   r.update(status='cooking',cook_start=t,cook_wait=t-r['order_time'],cook_minutes=float(duration),worker=worker)
   schedule(t+duration,'food_ready',i,(c,worker));log(t,'cook_start',i,counter=c,worker=worker)
 def seated(i,t,seat=None):
  r=records[i];j=r['chosen'];r.update(status='served',served=True,seat_time=t,seat_wait=t-r['food_ready'],total_wait=t-r['arrival'],taste=float(fit[i,j]),affordability=float(max(0.,1.-prices[j]/budgets[i])),wait_utility=s.wait_scale/(s.wait_scale+t-r['arrival']))
  r['missed_class']=bool(t>r['arrival']+class_gap[i]);log(t,'served',i,counter=r['counter'])
  if seat is not None:schedule(t+float(rng_for(day_seed,'dining',i).uniform(12.,25.)),'seat_release',i,seat)
 def dispatch_seats(t):
  while free_seats and seat_queue:seated(seat_queue.popleft(),t,heapq.heappop(free_seats))
 for i,arrival in enumerate(arrivals):schedule(arrival,'arrival',i)
 last_time=-1.
 while events:
  t,_,kind,i,payload=heapq.heappop(events)
  if t<last_time:raise AssertionError('time reversal')
  last_time=t;r=records[i]
  if kind=='arrival':
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
   dispatch_kiosks(t)
  elif kind=='food_ready':
   c,worker=payload;busy[c][worker]=False;expected_end[c][worker]=0.;r.update(status='waiting_seat',food_ready=t);log(t,'food_ready',i,counter=c,worker=worker);dispatch_kitchen(c,t)
   if court.seats:seat_queue.append(i);dispatch_seats(t)
   else:seated(i,t)
  elif kind=='seat_release':heapq.heappush(free_seats,payload);dispatch_seats(t)
 terminal={'served','no_feasible_menu','closed','closed_after_decision','outside_option','predicted_wait_decline','queue_abandonment','sold_out_during_choice'}
 if any(r['status'] not in terminal for r in records):raise AssertionError('unfinished customer')
 for c in range(len(stock)):
  for k,v in stock[c].items():
   if v<0 or not math.isclose(court.initial_stock[c][k],v+consumed[c][k]):raise AssertionError('inventory conservation')
 served=[r for r in records if r['served']];den=max(1,n);count=len(served)
 avg=lambda key:sum(r[key] for r in served)/max(1,count)
 revenue=sum(r['revenue'] for r in records);food_cost=sum(r['food_cost'] for r in records)
 leftover_cost=sum(q*court.ingredient_costs[k] for x in stock for k,q in x.items())
 waste_cost=s.perishable_fraction*leftover_cost;profit=revenue-food_cost-waste_cost
 result=dict(visitors=n,served=count,served_fraction=count/den,taste_per_buyer=avg('taste'),mean_total_wait=avg('total_wait'),mean_kiosk_wait=avg('kiosk_wait'),mean_decision_time=avg('decision'),mean_cook_wait=avg('cook_wait'),mean_cook_minutes=avg('cook_minutes'),mean_seat_wait=avg('seat_wait'),mean_inspected=sum(r['inspected'] for r in records)/den,revenue=revenue,food_cost=food_cost,waste_cost=waste_cost,contribution_profit=profit,profit_per_visitor=profit/den,store=profit/(den*s.profit_scale),kiosk_busy_minutes=kiosk_busy,kiosk_utilization=kiosk_busy/(court.kiosks*court.duration),overtime_minutes=max(0.,max((r['food_ready'] for r in served),default=court.duration)-court.duration),accepted_recommendations=sum(r['accepted'] for r in records),missed_class=sum(r.get('missed_class',False) for r in served),active_menus=m,menu_count=m)
 for status in sorted(terminal):result['count_'+status]=sum(r['status']==status for r in records)
 for name,mask in [('low_budget',budgets<(s.budget_low+s.budget_high)/2),('high_wait_sensitivity',ws>.7)]:
  group=[r for i,r in enumerate(records) if mask[i]]; size=len(group)
  result['visitors_'+name]=size
  result['service_'+name]=sum(r['served'] for r in group)/max(1,size)
 result['counter_revenue']=[sum(r['revenue'] for r in records if r['counter']==c) for c in range(len(court.servers))]
 return result,dict(records=records,trace=trace,initial_stock=court.initial_stock,consumed=consumed,leftover=stock) if raw else None
