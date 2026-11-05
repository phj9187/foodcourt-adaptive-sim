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
from .policy import ActorCritic, Collector, update_ppo
from .dashboard import comparison
from .inventory import Inventory
from .venues import coverage


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


def period(court, settings, objective, days, namespace, visitors, model=None,
           learn=False, changes=False):
    """Calendar-day service blocks with FEFO stock and lagged procurement.

    Each day's queues close after service, no overnight customer queues. No
    repeat-customer satisfaction/return behaviour is claimed. Menu changes are
    scheduled exogenously and identically in both arms.
    """
    court = copy.deepcopy(court)
    params=court.provenance.get('parameters')
    if not params:raise ValueError('research venue parameters are required')
    median_price=float(np.median([d.price for d in court.dishes]))
    settings=replace(settings,openness=min(1.,settings.openness*params['openness']/.65),
        attention_limit=params['attention_limit'],patience_minutes=params['patience_minutes'],
        taste_noise=params['taste_noise'],demand_cv=params['demand_cv'],
        decision_base=params['order_minutes']*.6,payment_minutes=params['order_minutes']*.4,
        dining_minutes=params['dining_minutes'],peak_fraction=params['peak_fraction'],
        venue_type=court.provenance['venue_type'],budget_low=.8*median_price,budget_high=1.6*median_price)
    inventory=Inventory(court,settings.seed,namespace)
    expected_visitors=court.provenance['expected_visitors'] if visitors is None else visitors
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
    rows = []
    for day in range(days):
        # A visible structural shift: held-out menus arrive at midpoint; one old
        # menu per counter is removed. IDs never enter the neural observation.
        if changes and day < max(1, days//2):
            court.dishes = [d for d in original if d not in held_out and d.counter!=len(court.servers)-1]
        elif changes:
            removed = {next((d.id for d in original if d.counter==c and d not in held_out), '')
                       for c in range(len(court.servers))}
            court.dishes = [d for d in original if d.id not in removed]
        waste_cost=inventory.begin_day(day)
        court.initial_stock=inventory.stock()
        court.provenance['supply_observations']=inventory.observations(day)
        weekday=(calendar_start+day)%7
        opened=weekday in court.provenance['weekdays']
        court.provenance['weekday']=weekday
        court.provenance['days_to_next_service']=next(d for d in range(1,8) if (weekday+d)%7 in court.provenance['weekdays'])
        day_visitors=expected_visitors*(params['weekday_factor'] if weekday>=5 else 1.) if opened else 0.
        before = len(collector.samples) if collector else 0
        row, raw = simulate(court, settings, 'neural' if model is not None else 'none',
                            day=day, namespace=namespace, visitors=day_visitors,
                            raw=True, controller=collector,period_day=day,period_days=days)
        carried=inventory.finish_day(day,raw['consumed'],raw['initial_stock'],opened=opened)
        row['waste_cost']=waste_cost
        row['contribution_profit']=row['revenue']-row['food_cost']-waste_cost
        component=metrics(row,raw,settings.profit_scale)
        score=objective.score(component)
        customer_score,store_score=objective.components(component)
        if collector and len(collector.samples):
            collector.samples[-1]['reward'] += score
        rows.append(dict(day=day, weekday=weekday, opened=opened, score=score, customer_score=customer_score, store_score=store_score,
                         components=component, menus=len(court.dishes),
                         active_stores=len({d.counter for d in court.dishes}),
                         behaviour=dict(openness=settings.openness,patience=settings.patience_minutes,
                                        taste_noise=settings.taste_noise,demand_cv=settings.demand_cv),
                         decisions=(len(collector.samples)-before) if collector else 0,
                         visitors=row['visitors'], served=row['served'],
                         served_fraction=row['served_fraction'], wait=row['mean_total_wait'],
                         revenue=row['revenue'], contribution_profit=row['contribution_profit'],
                         waste_cost=row['waste_cost'], accepted=row['accepted_recommendations'],
                         kiosk_abandonment=row['count_queue_abandonment'],
                         outside_option=row['count_outside_option'],
                         no_feasible_menu=row['count_no_feasible_menu'],
                         inventory=dict(initial=raw['initial_stock'], consumed=raw['consumed'],
                                        leftover=raw['leftover'], carried=carried,
                                        supply=inventory.audit())))
    return rows, collector.samples if collector else []


def visitor_mean(court,config):
    return config['visitors'] if config['demand_mode']=='fixed' else court.provenance['expected_visitors']*config['demand_scale']


def evaluate(model, settings, objective, config, split, on_progress=None):
    paired = []
    for rep in range(config['evaluation_repeats']):
        for index in range(config[split+'_courts']):
            court = make_court(settings, index, split, config['visitors'],config[split+'_courts'])
            for changes in (False, True):
                namespace = f'{split}_rep{rep}_changes{int(changes)}'
                args = (court, settings, objective, config['period_days'], namespace, visitor_mean(court,config))
                baseline, _ = period(*args, changes=changes)
                ai, _ = period(*args, model=model, changes=changes)
                for b,a in zip(baseline, ai):
                    assert b['visitors']==a['visitors']
                bscore = float(np.mean([r['score'] for r in baseline]))
                ascore = float(np.mean([r['score'] for r in ai]))
                paired.append(dict(court=court.id, repeat=rep, changes=changes,
                                   baseline_score=bscore, ai_score=ascore, delta=ascore-bscore,
                                   customer_delta=float(np.mean([a['customer_score']-b['customer_score']
                                                               for b,a in zip(baseline,ai)])),
                                   profit_delta=float(np.mean([a['contribution_profit']-b['contribution_profit']
                                                             for b,a in zip(baseline,ai)])),
                                   # Relative percentage omitted near zero and for signed scores.
                                   baseline=baseline, ai=ai))
                if on_progress:
                    on_progress(len(paired),config['evaluation_repeats']*config[split+'_courts']*2)
    return paired


def summarize(pairs):
    result = {}
    for changes in (False, True):
        selected = [p for p in pairs if p['changes']==changes]
        # Independent courts, NOT correlated days/repeats, are the unit of uncertainty.
        grouped = {}
        for p in selected:
            grouped.setdefault(p['court'], []).append(p['delta'])
        values = np.array([np.mean(v) for v in grouped.values()])
        result['changed' if changes else 'stable'] = dict(
            courts=len(values), mean_delta=float(values.mean()),
            mean_customer_delta=float(np.mean([p['customer_delta'] for p in selected])),
            mean_profit_delta_won=float(np.mean([p['profit_delta'] for p in selected])),
            court_sd=float(values.std(ddof=1)) if len(values)>1 else None,
            harm_fraction=float(np.mean(values<0)),
            descriptive_95_half_width=float(1.96*values.std(ddof=1)/np.sqrt(len(values)))
                if len(values)>1 else None,
            note='normal approximation; small-court pilot is not inferential evidence')
    return result


DEFAULT = dict(seed=20261105, lhs_samples=8, powers=[1.0], replicas=3,
               train_courts=12, validation_courts=4, test_courts=8,
               training_blocks=48, period_days=5, visitors=150,
               evaluation_repeats=3, validation_every=12, ppo_epochs=4,
               hidden=48, learning_rate=.0003, threads=1,venue_types=['campus'],
               demand_mode='capacity',demand_scale=1.,venue_lhs_candidates=24)


def run(config, output):
    cfg = dict(DEFAULT, **config)
    if set(cfg)-set(DEFAULT):
        raise ValueError('unknown configuration keys: '+str(set(cfg)-set(DEFAULT)))
    for key in ('seed','lhs_samples','replicas','train_courts','validation_courts','test_courts',
                'training_blocks','period_days','visitors','evaluation_repeats','validation_every',
                'ppo_epochs','hidden','threads'):
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
    if cfg['demand_mode'] not in ('capacity','fixed') or not 0<cfg['demand_scale']<=10:
        raise ValueError('invalid demand mode/scale')
    if type(cfg['venue_lhs_candidates']) is not int or not 1<=cfg['venue_lhs_candidates']<=100:
        raise ValueError('venue_lhs_candidates 1..100 required')
    if any(cfg[k]<len(cfg['venue_types']) for k in ('train_courts','validation_courts','test_courts')):
        raise ValueError('each split requires at least one venue per selected type')
    torch.set_num_threads(cfg['threads'])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    settings = Settings(seed=cfg['seed'], recommendation_count=1,venue_types=tuple(cfg['venue_types']),venue_lhs_candidates=cfg['venue_lhs_candidates'])
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
    manifest = dict(schema=1, config=cfg, code_sha256=fingerprint.hexdigest(),
                    objectives=[asdict(o) for o in objectives], lhs_cube=cube,
                    versions=dict(torch=torch.__version__, numpy=np.__version__),
                    assumptions=['synthetic customer response, not measured satisfaction',
                    'five-parameter objective family; powers apply to daily visitor-average components',
                    'daily FEFO expiry and lagged procurement; service queues drain before next service, no overnight queue carry',
                    'no mandatory subgroup penalties; no forced benefit of recommendation',
                    'test frozen policy; local adaptation must be evaluated in a separate future protocol'])
    target = output/'manifest.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('existing experiment configuration/source changed; choose a new output folder')
    atomic_json(target, manifest)
    catalog={split:[make_court(settings,i,split,cfg['visitors'],cfg[split+'_courts'])
                    for i in range(cfg[split+'_courts'])] for split in ('train','validation','test')}
    atomic_json(output/'court_catalog.json',{split:[asdict(c) for c in courts] for split,courts in catalog.items()})
    atomic_json(output/'venue_coverage.json',{split:coverage(courts) for split,courts in catalog.items()})
    summaries = []
    for objective in objectives:
        for rep in range(cfg['replicas']):
            job = output/objective.id/f'rep_{rep}'; job.mkdir(parents=True,exist_ok=True)
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
                        evaluation_total=cfg['evaluation_repeats']*cfg['validation_courts']*2)
                initial=evaluate(model,settings,objective,cfg,'validation',
                                 lambda n,total:publish('validation',0,evaluated=n,evaluation_total=total))
                live_history.append(dict(block=0,**comparison(initial)))
                publish('training',0)
            for block in range(start,cfg['training_blocks']):
                publish('training',block)
                # Revisit each court in shuffled cycles; retain multi-day continuity inside blocks.
                cycle = block//cfg['train_courts']
                order = rng_for(seed,'order',cycle).permutation(cfg['train_courts'])
                index = int(order[block%cfg['train_courts']])
                court = catalog['train'][index]
                rows, samples = period(court,settings,objective,cfg['period_days'],
                    f'train_rep{rep}_block{block}',visitor_mean(court,cfg),model,learn=True,changes=bool(block%2))
                info = update_ppo(model,optimizer,samples,epochs=cfg['ppo_epochs'])
                history.append(dict(block=block+1,court=court.id,score=float(np.mean([r['score'] for r in rows])),**info))
                if (block+1)%cfg['validation_every']==0 or block+1==cfg['training_blocks']:
                    publish('validation',block+1,evaluated=0,
                            evaluation_total=cfg['evaluation_repeats']*cfg['validation_courts']*2)
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
                    evaluation_total=cfg['evaluation_repeats']*cfg['test_courts']*2)
            pairs = evaluate(model,settings,objective,cfg,'test',
                lambda n,total:publish('testing',cfg['training_blocks'],evaluated=n,evaluation_total=total))
            summary = dict(objective=asdict(objective),replica=rep,validation_delta=best_score,
                           results=summarize(pairs))
            atomic_json(job/'paired_test.json',pairs); atomic_json(job/'summary.json',summary)
            atomic_json(job/'learning_history.json',history)
            publish('complete',cfg['training_blocks'],
                    test_stable=comparison([p for p in pairs if not p['changes']]),
                    test_changed=comparison([p for p in pairs if p['changes']]))
            summaries.append(summary); atomic_json(output/'summary.json',summaries)
    export_csv(output, summaries)
    return summaries


def export_csv(output, summaries):
    import csv
    rows = []
    for summary in summaries:
        for scenario, values in summary['results'].items():
            rows.append(dict(**summary['objective'], replica=summary['replica'],
                             scenario=scenario, **values))
    with (Path(output)/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}, args.output)


if __name__=='__main__':
    main()
