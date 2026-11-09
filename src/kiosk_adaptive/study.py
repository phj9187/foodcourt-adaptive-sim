"""Paired objective-specific experiments, resumable at each court-period boundary."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import copy
import numpy as np
import torch
import yaml
from .models import Settings, rng_for, seed_for
from .design import lhs_objectives, make_court, metrics
from .environment import simulate
from .policy import ActorCritic, Collector, RuleController, update_ppo
from .dashboard import comparison
from .inventory import Inventory, demand_stock
from .venues import coverage
from .customers import CustomerMemory,WORLDS


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path); tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    if path.exists():
        previous_tmp = path.with_suffix('.previous.tmp')
        shutil.copyfile(path, previous_tmp)
        os.replace(previous_tmp, path.with_suffix('.previous.pt'))
    os.replace(tmp, path)


def customer_reward(record, objective, visitors, profit_scale):
    """One customer's exactly attributable share of the daily F (power-1 terms).

    Taste/affordability/wait and revenue minus food cost are sums over customers,
    so they split exactly. Waste, abandonment externalities and the power
    correction are not, and are paid as the day-end residual.
    """
    den=max(1,visitors);o=objective
    customer=(o.taste*record['taste']+o.affordability*record['affordability']+o.waiting*record['wait_utility'])/den
    store=o.profit_share*(record['revenue']-record['food_cost'])/(den*profit_scale)
    return o.customer_share*customer+(1-o.customer_share)*store


def period_score(rows, key='score'):
    """Mean over service days; closed days carry no customers and no score."""
    values=[r[key] for r in rows if r['opened']]
    return float(np.mean(values)) if values else 0.


def period(court, settings, objective, days, namespace, visitors, model=None,
           learn=False, changes=False, policy=None, reward_mode='decomposed'):
    """Calendar-day service blocks with FEFO stock and lagged procurement.

    Each day's queues close after service, no overnight customer queues. A
    repeat-customer experience is carried inside a block only. Menu changes are
    scheduled exogenously and identically in both arms.
    """
    court = copy.deepcopy(court)
    params=court.provenance.get('parameters')
    if not params:raise ValueError('research venue parameters are required')
    settings=replace(settings,openness=min(1.,settings.openness*params['openness']/.65),
        attention_limit=params['attention_limit'],patience_minutes=params['patience_minutes'],
        taste_noise=params['taste_noise'],demand_cv=params['demand_cv'],
        decision_base=params['order_minutes']*.6,payment_minutes=params['order_minutes']*.4,
        dining_minutes=params['dining_minutes'],peak_fraction=params['peak_fraction'],
        venue_type=court.provenance['venue_type'],repeat_share=params['repeat_share'],
        memory_strength=params['memory_strength'])
    if settings.taste_noise_override is not None:
        settings=replace(settings,taste_noise=settings.taste_noise_override)
    if reward_mode not in ('daily','decomposed'):raise ValueError('unknown reward mode')
    expected_visitors=court.provenance['expected_visitors'] if visitors is None else visitors
    if settings.stock_basis=='demand':
        court.initial_stock=demand_stock(court,expected_visitors,settings)
    inventory=Inventory(court,settings.seed,namespace)
    memory=CustomerMemory()
    original = list(court.dishes)
    held_out = [d for d in original if int(d.id.rsplit('M',1)[1]) % 3 == 0]
    if not held_out:
        held_out = original[:1]
    for c in range(len(court.servers)):
        pool = [d for d in original if d.counter==c]
        if all(d in held_out for d in pool):
            held_out.remove(pool[-1])
    collector = Collector(model, deterministic=not learn) if model is not None else None
    calendar_start=seed_for(settings.seed,court.id,namespace,'calendar')%7
    rows = [];days_data=[];carried_waste=0.
    for day in range(days):
        # A visible structural shift: held-out menus arrive at midpoint; one old
        # menu per counter is removed. IDs never enter the neural observation.
        if changes and day < max(1, days//2):
            court.dishes = [d for d in original if d not in held_out and d.counter!=len(court.servers)-1]
        elif changes:
            removed = {next((d.id for d in original if d.counter==c and d not in held_out), '')
                       for c in range(len(court.servers))}
            court.dishes = [d for d in original if d.id not in removed]
        expired=inventory.begin_day(day)
        court.initial_stock=inventory.stock()
        court.provenance['supply_observations']=inventory.observations(day)
        weekday=(calendar_start+day)%7
        opened=weekday in court.provenance['weekdays']
        court.provenance['weekday']=weekday
        court.provenance['days_to_next_service']=next(d for d in range(1,8) if (weekday+d)%7 in court.provenance['weekdays'])
        day_visitors=expected_visitors*(params['weekday_factor'] if weekday>=5 else 1.) if opened else 0.
        before = len(collector.samples) if collector else 0
        row, raw = simulate(court, settings, policy or ('neural' if model is not None else 'none'),
                            day=day, namespace=namespace, visitors=day_visitors,
                            raw=True, controller=collector,period_day=day,period_days=days,memory=memory)
        carried=inventory.finish_day(day,raw['consumed'],raw['initial_stock'],opened=opened,lost=raw['lost'])
        # Expiry on closed days is charged to the next service day, so it is
        # normalized by real visitors instead of a placeholder denominator of 1.
        carried_waste+=expired
        waste_cost=carried_waste if opened else 0.
        if opened:carried_waste=0.
        row['waste_cost']=waste_cost
        row['contribution_profit']=row['revenue']-row['food_cost']-waste_cost
        days_data.append((raw,before,len(collector.samples) if collector else 0))
        rows.append(dict(day=day, weekday=weekday, opened=opened, score=0., customer_score=0., store_score=0.,
                         expired_today=expired,stock_blocked=row['stock_blocked'],
                         components=None, menus=len(court.dishes),
                         active_stores=len({d.counter for d in court.dishes}),
                         behaviour=dict(openness=settings.openness,patience=settings.patience_minutes,
                                        taste_noise=settings.taste_noise,demand_cv=settings.demand_cv,
                                        world=settings.behaviour_world,dominant_choice_share=settings.dominant_choice_share,
                                        repeat_share=settings.repeat_share,memory_strength=settings.memory_strength),
                         decisions=(len(collector.samples)-before) if collector else 0,
                         visitors=row['visitors'], served=row['served'],
                         potential_visitors=row['potential_visitors'],actual_visitors=row['actual_visitors'],
                         repeat_opportunities=row['repeat_opportunities'],
                         visit_declined=row['count_visit_declined'],queue_balk=row['count_queue_balk'],
                         paid_cancellations=sum(row['count_'+stage] for stage in ('cancelled_waiting_kitchen','cancelled_cooking','cancelled_waiting_seat')),
                         refunds=row['refunds'],prepared_food_loss=row['prepared_food_loss'],
                         wait_p90=row['wait_p90'],wait_p95=row['wait_p95'],
                         recommendation_exposures=row['recommendation_exposures'],
                         choice_world_counts=row['choice_world_counts'],
                         subgroups={name:dict(opportunities=row['visitors_'+name],served_fraction=row['service_'+name]) for name in ('low_budget','high_wait_sensitivity')},
                         served_fraction=row['served_fraction'], wait=row['mean_total_wait'],
                         revenue=row['revenue'], food_cost=row['food_cost'], contribution_profit=row['contribution_profit'],
                         waste_cost=row['waste_cost'], accepted=row['accepted_recommendations'],
                         kiosk_abandonment=row['count_queue_abandonment'],
                         outside_option=row['count_outside_option'],
                         no_feasible_menu=row['count_no_feasible_menu'],
                         inventory=dict(initial=raw['initial_stock'], consumed=raw['consumed'],
                                        leftover=raw['leftover'], carried=carried,
                                        supply=inventory.audit())))
    opened_rows=[k for k,r in enumerate(rows) if r['opened']]
    if carried_waste>0 and opened_rows:
        # Expiry after the last service day is charged to that day.
        last=rows[opened_rows[-1]];last['waste_cost']+=carried_waste
        last['contribution_profit']-=carried_waste;carried_waste=0.
    for k,(r,(raw,start,end)) in enumerate(zip(rows,days_data)):
        if not r['opened']:
            r['components']=dict(taste=0.,affordability=0.,waiting=0.,margin=0.,waste=0.)
            continue
        component=metrics(r,raw,settings.profit_scale)
        r['components']=component;r['score']=objective.score(component)
        r['customer_score'],r['store_score']=objective.components(component)
        if not collector or not collector.samples:continue
        if reward_mode=='decomposed' and end>start:
            paid=0.
            for sample in collector.samples[start:end]:
                value=customer_reward(raw['records'][sample['person']],objective,r['visitors'],settings.profit_scale)
                sample['reward']+=value;paid+=value
            # Residual keeps the per-day reward sum exactly equal to F.
            collector.samples[end-1]['reward']+=r['score']-paid
        else:
            target=end-1 if end>start else max(0,start-1)
            collector.samples[target]['reward']+=r['score']
    if carried_waste>0:
        rows[-1]['unattributed_waste_cost']=carried_waste
    return rows, collector.samples if collector else []


def visitor_mean(court,config):
    if config['demand_mode']=='fixed':return config['visitors']
    if config['demand_mode']=='exogenous':return config['visitors']*court.provenance['exogenous_demand_factor']*config['demand_scale']
    return court.provenance['expected_visitors']*config['demand_scale']


def evaluate(model, settings, objective, config, split, on_progress=None):
    paired = []
    worlds=config.get('behaviour_worlds',list(WORLDS))
    total=config['evaluation_repeats']*config[split+'_courts']*2*len(worlds)
    # Controls are a final-test diagnostic; validation only selects checkpoints.
    controls_on=split=='test'
    for rep in range(config['evaluation_repeats']):
        for index in range(config[split+'_courts']):
            court = make_court(settings, index, split, config['visitors'],config[split+'_courts'])
            for world in worlds:
                for changes in (False, True):
                    namespace = f'{split}_rep{rep}_changes{int(changes)}'
                    args = (court, replace(settings,behaviour_world=world), objective, config['period_days'], namespace, visitor_mean(court,config))
                    baseline, _ = period(*args, changes=changes)
                    ai, _ = period(*args, model=model, changes=changes)
                    for b,a in zip(baseline, ai):
                        # Attendance may change with experience. Pair external
                        # opportunities, never force equal actual return visits.
                        assert b['potential_visitors']==a['potential_visitors']
                    bscore = period_score(baseline)
                    ascore = period_score(ai)
                    controls={}
                    if controls_on and config.get('comparison_controls',True):
                        for arm in ('random','popular'):
                            rows,_=period(*args,changes=changes,policy=arm)
                            value=period_score(rows)
                            controls[arm]=dict(score=value,delta_vs_none=value-bscore,
                                ai_minus_control=ascore-value,rows=rows)
                    if controls_on and config.get('rule_controls',True):
                        for arm in RULE_ARMS:
                            rule=RuleController(arm,objective,settings.wait_scale,settings.profit_scale)
                            rows,_=period(*args,model=rule,changes=changes)
                            value=period_score(rows)
                            # Rows are not stored for rules to keep result files small.
                            controls[arm]=dict(score=value,delta_vs_none=value-bscore,ai_minus_control=ascore-value,
                                customer_delta=period_score(rows,'customer_score')-period_score(baseline,'customer_score'))
                    paired.append(dict(court=court.id, repeat=rep, changes=changes,behaviour_world=world,
                                       baseline_score=bscore, ai_score=ascore, delta=ascore-bscore,
                                       customer_delta=period_score(ai,'customer_score')-period_score(baseline,'customer_score'),
                                       profit_delta=float(np.mean([a['contribution_profit']-b['contribution_profit']
                                                                 for b,a in zip(baseline,ai) if b['opened']] or [0.])),
                                       controls=controls,baseline=baseline, ai=ai))
                    if on_progress:on_progress(len(paired),total)
    return paired


RULE_ARMS=('rule_fit','rule_fast','rule_margin','rule_objective')
# Two-sided 97.5% Student t quantiles for small numbers of independent courts.
T975={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,
      13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,25:2.060,30:2.042,40:2.021,60:2.000}


def t975(df):
    keys=[k for k in sorted(T975) if k<=df]
    return T975[keys[-1]] if keys else T975[1]


def interval(values, seed=0, resamples=5000):
    """Court-level t interval and percentile bootstrap; descriptive for few courts."""
    values=np.asarray(values,dtype=float);n=len(values)
    out=dict(courts=n,mean=float(values.mean()) if n else None,court_sd=float(values.std(ddof=1)) if n>1 else None,
             t95=None,bootstrap95=None)
    if n>1:
        half=t975(n-1)*values.std(ddof=1)/np.sqrt(n)
        out['t95']=[float(values.mean()-half),float(values.mean()+half)]
        rng=np.random.default_rng(seed)
        means=values[rng.integers(0,n,(resamples,n))].mean(axis=1)
        out['bootstrap95']=[float(x) for x in np.quantile(means,[.025,.975])]
    return out


def court_means(pairs, key=lambda p:p['delta']):
    grouped={}
    for p in pairs:grouped.setdefault(p['court'],[]).append(key(p))
    return {court:float(np.mean(v)) for court,v in grouped.items()}


def summarize(pairs, seed=0):
    result = {}
    for changes in (False, True):
        selected = [p for p in pairs if p['changes']==changes]
        if not selected:continue
        # Independent courts, NOT correlated days/repeats/worlds, are the unit of uncertainty.
        by_court=court_means(selected)
        values = np.array(list(by_court.values()))
        stats=interval(values,seed)
        entry = dict(
            courts=len(values), mean_delta=float(values.mean()),
            mean_customer_delta=float(np.mean([p['customer_delta'] for p in selected])),
            mean_profit_delta_won=float(np.mean([p['profit_delta'] for p in selected])),
            mean_actual_visit_delta=float(np.mean([np.mean([a['actual_visitors']-b['actual_visitors'] for b,a in zip(p['baseline'],p['ai'])]) for p in selected])),
            mean_served_delta=float(np.mean([np.mean([a['served']-b['served'] for b,a in zip(p['baseline'],p['ai'])]) for p in selected])),
            court_sd=stats['court_sd'],harm_fraction=float(np.mean(values<0)),
            t95=stats['t95'],bootstrap95=stats['bootstrap95'],court_deltas=by_court,
            note='court-level t and bootstrap intervals; few courts means descriptive, not confirmatory')
        arms=sorted({arm for p in selected for arm in p.get('controls',{})})
        if arms:
            entry['controls']={arm:dict(mean_delta_vs_none=float(np.mean([p['controls'][arm]['delta_vs_none'] for p in selected])),
                                        mean_ai_minus_control=float(np.mean([p['controls'][arm]['ai_minus_control'] for p in selected])))
                               for arm in arms}
            rules=[a for a in arms if a.startswith('rule_')]
            if rules:
                best=max(rules,key=lambda a:entry['controls'][a]['mean_delta_vs_none'])
                gap=court_means(selected,lambda p:p['controls'][best]['ai_minus_control'])
                entry['learning_check']=dict(best_rule=best,best_rule_delta=entry['controls'][best]['mean_delta_vs_none'],
                    ai_minus_best_rule=interval(list(gap.values()),seed),
                    verdict='ai_beats_best_rule' if np.mean(list(gap.values()))>0 else 'ai_not_better_than_best_rule',
                    note='best rule chosen on the same test pairs, so this bar is slightly optimistic for the rule')
        result['changed' if changes else 'stable'] = entry
    return result


def combine_replicas(summaries, seed=0):
    """Pool independently initialized replicas per objective: average each court over replicas."""
    combined={}
    for objective_id in sorted({s['objective']['id'] for s in summaries}):
        group=[s for s in summaries if s['objective']['id']==objective_id]
        entry=dict(objective=group[0]['objective'],replicas=len(group))
        for scenario in ('stable','changed'):
            courts={}
            for s in group:
                for court,value in s['results'].get(scenario,{}).get('court_deltas',{}).items():
                    courts.setdefault(court,[]).append(value)
            if courts:
                values=[float(np.mean(v)) for v in courts.values()]
                entry[scenario]=dict(**interval(values,seed),harm_fraction=float(np.mean(np.array(values)<0)),
                    replica_means=[s['results'][scenario]['mean_delta'] for s in group if scenario in s['results']])
        combined[objective_id]=entry
    return combined


def baseline_audit(catalog, settings, cfg, objective, thresholds):
    """No-recommendation sanity check of every venue before any AI comparison."""
    report={}
    for split,courts in catalog.items():
        for court in courts:
            rows,_=period(court,settings,objective,cfg['period_days'],f'audit_{split}',visitor_mean(court,cfg))
            opened=[r for r in rows if r['opened']]
            potential=sum(r['visitors'] for r in opened);served=sum(r['served'] for r in opened)
            revenue=sum(r['revenue'] for r in opened);waste=sum(r['waste_cost'] for r in rows)
            item=dict(split=split,served_fraction=served/max(1,potential),
                      stock_blocked_fraction=sum(r['stock_blocked'] for r in opened)/max(1,potential),
                      waste_to_revenue=waste/max(1.,revenue),
                      zero_service_days=sum(r['visitors']>0 and r['served']==0 for r in opened),
                      max_abs_daily_score=max((abs(r['score']) for r in opened),default=0.))
            flags=[]
            if item['served_fraction']<thresholds['min_served_fraction']:flags.append('low_service')
            if item['stock_blocked_fraction']>thresholds['max_stock_blocked_fraction']:flags.append('stockouts')
            if item['waste_to_revenue']>thresholds['max_waste_to_revenue']:flags.append('waste')
            if item['zero_service_days']>thresholds['max_zero_service_days']:flags.append('zero_service_days')
            item['flags']=flags;report[court.id]=item
    return dict(thresholds=thresholds,courts=report,
                flagged=sorted(k for k,v in report.items() if v['flags']),
                note='baseline realism gate; flagged venues are reported, interpret their AI effects with caution')


DEFAULT = dict(seed=20261105, lhs_samples=8, powers=[1.0], replicas=3,
               train_courts=12, validation_courts=4, test_courts=8,
               training_blocks=48, period_days=5, visitors=150,
               evaluation_repeats=3, validation_every=12, ppo_epochs=4,
               hidden=48, learning_rate=.0003, threads=1,venue_types=['campus'],
               demand_mode='exogenous',demand_scale=1.,venue_lhs_candidates=24,
               behaviour_worlds=list(WORLDS),comparison_controls=True,
               rule_controls=True,reward_mode='decomposed',gae_lambda=.95,
               taste_noise_override=None,input_noise=.06,input_missing_probability=0.,input_minutes=0.,
               stock_basis='demand',baseline_audit=True,workers=1,
               audit_thresholds=dict(min_served_fraction=.6,max_stock_blocked_fraction=.15,
                                     max_waste_to_revenue=.5,max_zero_service_days=0))


def run_job(objective, rep, cfg, settings, catalog, audit, output):
    """Train, select and test one (objective, replica); safe to run in a worker process."""
    torch.set_num_threads(cfg['threads'])
    job = output/objective.id/f'rep_{rep}'
    if (job/'summary.json').exists():
        return json.loads((job/'summary.json').read_text(encoding='utf-8'))
    job.mkdir(parents=True,exist_ok=True)
    seed = seed_for(cfg['seed'],objective.id,rep) % (2**31)
    torch.manual_seed(seed)
    model = ActorCritic(cfg['hidden'])
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    start, best_score, best, history = 0, -float('inf'), None, []
    live_history=[]
    checkpoint = job/'checkpoint.pt'
    if checkpoint.exists():
        try:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        except (RuntimeError, EOFError, OSError):
            saved = torch.load(checkpoint.with_suffix('.previous.pt'),map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, best_score, best, history = saved['block'], saved['best_score'], saved['best'], saved['history']
        live_history=saved.get('live_history',[])
        torch.set_rng_state(saved['torch_rng'])
    def publish(phase, block, **extra):
        atomic_json(job/'live.json',dict(phase=phase,block=block,
                    total=cfg['training_blocks'],history=live_history,**extra))
    if start==0 and not live_history:
        publish('validation',0,evaluated=0,
                evaluation_total=cfg['evaluation_repeats']*cfg['validation_courts']*2*len(cfg['behaviour_worlds']))
        initial=evaluate(model,settings,objective,cfg,'validation',
                         lambda n,total:publish('validation',0,evaluated=n,evaluation_total=total))
        live_history.append(dict(block=0,**comparison(initial)))
        # The untrained model is a legitimate candidate: if training only
        # hurts on validation, the initial network is kept.
        best_score=float(np.mean([p['delta'] for p in initial]));best=copy.deepcopy(model.state_dict())
        publish('training',0)
    for block in range(start,cfg['training_blocks']):
        publish('training',block)
        # Revisit each court in shuffled cycles; retain multi-day continuity inside blocks.
        cycle = block//cfg['train_courts']
        order = rng_for(seed,'order',cycle).permutation(cfg['train_courts'])
        index = int(order[block%cfg['train_courts']])
        court = catalog['train'][index]
        world=cfg['behaviour_worlds'][block%len(cfg['behaviour_worlds'])]
        rows, samples = period(court,replace(settings,behaviour_world=world),objective,cfg['period_days'],
            f'train_rep{rep}_block{block}',visitor_mean(court,cfg),model,learn=True,changes=bool(block%2),
            reward_mode=cfg['reward_mode'])
        info = update_ppo(model,optimizer,samples,epochs=cfg['ppo_epochs'],gae_lambda=cfg['gae_lambda'])
        history.append(dict(block=block+1,court=court.id,behaviour_world=world,score=period_score(rows),**info))
        if (block+1)%cfg['validation_every']==0 or block+1==cfg['training_blocks']:
            publish('validation',block+1,evaluated=0,
                    evaluation_total=cfg['evaluation_repeats']*cfg['validation_courts']*2*len(cfg['behaviour_worlds']))
            validation = evaluate(model,settings,objective,cfg,'validation',
                lambda n,total:publish('validation',block+1,evaluated=n,evaluation_total=total))
            value = float(np.mean([p['delta'] for p in validation]))
            history[-1]['validation_delta'] = value
            live_history.append(dict(block=block+1,**comparison(validation)))
            if value>best_score:
                best_score=value; best=copy.deepcopy(model.state_dict())
        atomic_torch(checkpoint,dict(block=block+1,model=model.state_dict(),
            optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
            best_score=best_score,best=best,history=history,live_history=live_history))
        atomic_json(job/'progress.json',dict(block=block+1,total=cfg['training_blocks'],last=history[-1]))
        publish('training',block+1)
        print(f'{objective.id} rep={rep} block={block+1}/{cfg["training_blocks"]}',flush=True)
    model.load_state_dict(best)
    publish('testing',cfg['training_blocks'],evaluated=0,
            evaluation_total=cfg['evaluation_repeats']*cfg['test_courts']*2*len(cfg['behaviour_worlds']))
    pairs = evaluate(model,settings,objective,cfg,'test',
        lambda n,total:publish('testing',cfg['training_blocks'],evaluated=n,evaluation_total=total))
    arms=sorted({arm for p in pairs for arm in p['controls']})
    summary = dict(objective=asdict(objective),replica=rep,validation_delta=best_score,
                   validation_note='maximum over repeated checkpoint evaluations on fixed validation venues; optimistic, report test results',
                   results=summarize(pairs,seed),
                   by_behaviour_world={world:summarize([p for p in pairs if p['behaviour_world']==world],seed) for world in cfg['behaviour_worlds']},
                   control_comparisons={arm:dict(mean_delta_vs_none=float(np.mean([p['controls'][arm]['delta_vs_none'] for p in pairs])),
                       mean_ai_minus_control=float(np.mean([p['controls'][arm]['ai_minus_control'] for p in pairs]))) for arm in arms},
                   flagged_test_courts=[c for c in (audit or {}).get('flagged',[]) if c.startswith('test_')])
    atomic_json(job/'paired_test.json',pairs); atomic_json(job/'summary.json',summary)
    atomic_json(job/'learning_history.json',history)
    publish('complete',cfg['training_blocks'],
            test_stable=comparison([p for p in pairs if not p['changes']]),
            test_changed=comparison([p for p in pairs if p['changes']]))
    return summary


def run(config, output):
    cfg = dict(DEFAULT, **config)
    if set(cfg)-set(DEFAULT):
        raise ValueError('unknown configuration keys: '+str(set(cfg)-set(DEFAULT)))
    for key in ('seed','lhs_samples','replicas','train_courts','validation_courts','test_courts',
                'training_blocks','period_days','visitors','evaluation_repeats','validation_every',
                'ppo_epochs','hidden','threads','workers'):
        if type(cfg[key]) is not int or cfg[key] < (0 if key=='seed' else 1):
            raise ValueError('invalid '+key)
    if not cfg['powers'] or any(not math.isfinite(p) or p<=0 for p in cfg['powers']):
        raise ValueError('positive powers required')
    if len(set(cfg['powers']))!=len(cfg['powers']) or not 0<cfg['learning_rate']<1:
        raise ValueError('invalid powers/learning rate')
    if cfg['period_days']<2:
        raise ValueError('period_days >= 2 for change experiment')
    if not cfg['venue_types'] or len(set(cfg['venue_types']))!=len(cfg['venue_types']) or not set(cfg['venue_types'])<=set(('campus','mall','highway','airport','leisure')):
        raise ValueError('invalid venue_types')
    if cfg['demand_mode'] not in ('capacity','fixed','exogenous') or not 0<cfg['demand_scale']<=10:
        raise ValueError('invalid demand mode/scale')
    if type(cfg['venue_lhs_candidates']) is not int or not 1<=cfg['venue_lhs_candidates']<=100:
        raise ValueError('venue_lhs_candidates 1..100 required')
    if not cfg['behaviour_worlds'] or len(set(cfg['behaviour_worlds']))!=len(cfg['behaviour_worlds']) or not set(cfg['behaviour_worlds'])<=set(WORLDS):
        raise ValueError('invalid behaviour_worlds')
    for key in ('comparison_controls','rule_controls','baseline_audit'):
        if type(cfg[key]) is not bool:raise ValueError(key+' must be boolean')
    if cfg['reward_mode'] not in ('daily','decomposed'):raise ValueError('invalid reward_mode')
    if cfg['gae_lambda'] is not None and not 0<=cfg['gae_lambda']<=1:raise ValueError('gae_lambda must be in [0,1] or null')
    cfg['audit_thresholds']=dict(DEFAULT['audit_thresholds'],**(cfg['audit_thresholds'] or {}))
    if any(cfg[k]<len(cfg['venue_types']) for k in ('train_courts','validation_courts','test_courts')):
        raise ValueError('each split requires at least one venue per selected type')
    torch.set_num_threads(cfg['threads'])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    settings = Settings(seed=cfg['seed'], recommendation_count=1,venue_types=tuple(cfg['venue_types']),venue_lhs_candidates=cfg['venue_lhs_candidates'],
                        taste_noise_override=cfg['taste_noise_override'],input_noise=cfg['input_noise'],
                        input_missing_probability=cfg['input_missing_probability'],input_minutes=cfg['input_minutes'],stock_basis=cfg['stock_basis']).validate()
    objectives, cube = lhs_objectives(cfg['lhs_samples'], cfg['seed'])
    objectives = [replace(o, id=o.id+f'_p{p:g}', power=p) for p in cfg['powers'] for o in objectives]
    fingerprint = hashlib.sha256()
    paths = list(Path(__file__).parent.glob('*.py')) + [
        Path(__file__).parent/'models.py',
        Path(__file__).parent/'data'/'menu_anchors.json',
        Path(__file__).parent/'data'/'sources.json',
        Path(__file__).parent/'data'/'venue_evidence.json']
    for path in sorted(paths):
        fingerprint.update(path.name.encode()); fingerprint.update(path.read_bytes())
    # workers only changes scheduling, never results, so it may differ on resume.
    manifest = dict(schema=2, config={k:v for k,v in cfg.items() if k!='workers'}, code_sha256=fingerprint.hexdigest(),
                    objectives=[asdict(o) for o in objectives], lhs_cube=cube,
                    versions=dict(torch=torch.__version__, numpy=np.__version__),
                    assumptions=['synthetic customer response, not measured satisfaction',
                    'five-parameter objective family; powers apply to daily visitor-average components',
                    'daily FEFO expiry and lagged procurement; service queues drain before next service, no overnight queue carry',
                    'no mandatory subgroup penalties; no forced benefit of recommendation',
                    'heterogeneous choice laws mixed within each day; synthetic customer traits, not calibrated coefficients',
                    'potential visit denominator includes policy-dependent nonattendance; visited and served are logged separately',
                    'repeat profiles and experience persist within a court block, reset between blocks',
                    'refund before cooking restores ingredients; prepared cancelled food cost is charged once in food_cost',
                    'capacity-based demand is a synthetic load design, frozen after court creation, not measured demand',
                    'test frozen policy; local adaptation must be evaluated in a separate future protocol'])
    target = output/'manifest.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('existing experiment configuration/source changed; choose a new output folder')
    atomic_json(target, manifest)
    catalog={split:[make_court(settings,i,split,cfg['visitors'],cfg[split+'_courts'])
                    for i in range(cfg[split+'_courts'])] for split in ('train','validation','test')}
    atomic_json(output/'court_catalog.json',{split:[asdict(c) for c in courts] for split,courts in catalog.items()})
    atomic_json(output/'venue_coverage.json',{split:coverage(courts) for split,courts in catalog.items()})
    audit=None
    if cfg['baseline_audit']:
        audit_path=output/'baseline_audit.json'
        if audit_path.exists():audit=json.loads(audit_path.read_text(encoding='utf-8'))
        else:
            audit=baseline_audit(catalog,settings,cfg,objectives[0],cfg['audit_thresholds'])
            atomic_json(audit_path,audit)
    jobs=[(objective,rep) for objective in objectives for rep in range(cfg['replicas'])]
    summaries=[None]*len(jobs)
    def record(index, summary):
        summaries[index]=summary
        done=[x for x in summaries if x is not None]
        atomic_json(output/'summary.json',done)
        atomic_json(output/'objective_summary.json',combine_replicas(done,cfg['seed']))
    if cfg['workers']==1:
        for index,(objective,rep) in enumerate(jobs):
            record(index,run_job(objective,rep,cfg,settings,catalog,audit,output))
    else:
        # Independent jobs share nothing but read-only inputs; results are
        # identical to sequential execution because every job seeds itself.
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(run_job,objective,rep,cfg,settings,catalog,audit,output):index
                     for index,(objective,rep) in enumerate(jobs)}
            for future in as_completed(futures):
                record(futures[future],future.result())
    export_csv(output, summaries)
    return summaries


def export_csv(output, summaries):
    import csv
    rows = []
    for summary in summaries:
        for scenario, values in summary['results'].items():
            flat={k:v for k,v in values.items() if not isinstance(v,(dict,list))}
            for key in ('t95','bootstrap95'):
                if values.get(key):flat[key+'_low'],flat[key+'_high']=values[key]
            check=values.get('learning_check')
            if check:flat.update(best_rule=check['best_rule'],best_rule_delta=check['best_rule_delta'],
                                 ai_minus_best_rule=check['ai_minus_best_rule']['mean'],learning_verdict=check['verdict'])
            rows.append(dict(**summary['objective'], replica=summary['replica'],
                             scenario=scenario, **flat))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (Path(output)/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields,restval='')
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,help='parallel (objective, replica) jobs; overrides the config')
    args = parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}
    if args.workers is not None:config['workers']=args.workers
    run(config, args.output)


if __name__=='__main__':
    main()
