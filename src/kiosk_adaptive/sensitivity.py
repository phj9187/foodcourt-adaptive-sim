"""Information-gap sensitivity: how much of a recommendation effect is an assumption.

Customers judge taste with venue-specific noise (taste_noise) while the kiosk
receives noisy, rounded self-reports (input_noise, missing inputs). Recommendation
benefits for customers come largely from that gap. This sweep re-runs the same
paired test venues under a grid of these assumptions for non-learning controls
(and optionally a trained PPO checkpoint), so reports can show the effect as a
function of the assumption instead of a single number.
"""
import argparse
import csv
from dataclasses import replace
import itertools
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from .customers import WORLDS
from .design import lhs_objectives, make_court
from .models import Settings
from .policy import ActorCritic, RuleController
from .study import RULE_ARMS, interval, period, period_score, visitor_mean, atomic_json

DEFAULT = dict(seed=20261105, visitors=150, period_days=7, test_courts=6, evaluation_repeats=1,
               demand_mode='exogenous', demand_scale=1., venue_types=['campus'], lhs_samples=8,
               objectives=['customer_only', 'profit_only', 'lhs_0000'], behaviour_worlds=list(WORLDS),
               taste_noise=[0., .1, .2, .45], input_noise=[.06, .3], input_missing_probability=[0.],
               arms=['random', 'popular', *RULE_ARMS], model_dir=None, hidden=48, threads=1)


def load_model(model_dir, objective_id, hidden):
    path = Path(model_dir)/objective_id/'rep_0'/'checkpoint.pt'
    if not path.exists():
        return None
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = ActorCritic(hidden); model.load_state_dict(saved['best'] or saved['model'])
    return model


def run(config, output):
    cfg = dict(DEFAULT, **config)
    unknown = set(cfg)-set(DEFAULT)
    if unknown:
        raise ValueError('unknown configuration keys: '+str(unknown))
    torch.set_num_threads(cfg['threads'])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'sensitivity_config.json', cfg)
    objectives = {o.id: o for o in lhs_objectives(cfg['lhs_samples'], cfg['seed'])[0]}
    chosen = [objectives[i] for i in cfg['objectives']]
    base = Settings(seed=cfg['seed'], venue_types=tuple(cfg['venue_types']))
    courts = [make_court(base, i, 'test', cfg['visitors'], cfg['test_courts']) for i in range(cfg['test_courts'])]
    models = {o.id: load_model(cfg['model_dir'], o.id+'_p1', cfg['hidden']) for o in chosen} if cfg['model_dir'] else {}
    rows = []
    for taste, noise, missing in itertools.product(cfg['taste_noise'], cfg['input_noise'], cfg['input_missing_probability']):
        settings = replace(base, taste_noise_override=taste, input_noise=noise, input_missing_probability=missing).validate()
        for objective in chosen:
            arms = list(cfg['arms'])+(['ppo'] if models.get(objective.id) is not None else [])
            deltas = {arm: {} for arm in arms}; customer = {arm: {} for arm in arms}
            for court in courts:
                for rep in range(cfg['evaluation_repeats']):
                    for world in cfg['behaviour_worlds']:
                        args = (court, replace(settings, behaviour_world=world), objective, cfg['period_days'],
                                f'test_rep{rep}_changes0', visitor_mean(court, cfg))
                        baseline, _ = period(*args)
                        b, bc = period_score(baseline), period_score(baseline, 'customer_score')
                        for arm in arms:
                            if arm in ('random', 'popular'):
                                result, _ = period(*args, policy=arm)
                            elif arm == 'ppo':
                                result, _ = period(*args, model=models[objective.id])
                            else:
                                result, _ = period(*args, model=RuleController(arm, objective, settings.wait_scale, settings.profit_scale))
                            deltas[arm].setdefault(court.id, []).append(period_score(result)-b)
                            customer[arm].setdefault(court.id, []).append(period_score(result, 'customer_score')-bc)
            for arm in arms:
                values = [float(np.mean(v)) for v in deltas[arm].values()]
                stats = interval(values, cfg['seed'])
                rows.append(dict(taste_noise=taste, input_noise=noise, input_missing_probability=missing,
                                 objective=objective.id, arm=arm, courts=stats['courts'], mean_delta=stats['mean'],
                                 t95_low=(stats['t95'] or [None, None])[0], t95_high=(stats['t95'] or [None, None])[1],
                                 bootstrap95_low=(stats['bootstrap95'] or [None, None])[0],
                                 bootstrap95_high=(stats['bootstrap95'] or [None, None])[1],
                                 mean_customer_delta=float(np.mean([np.mean(v) for v in customer[arm].values()]))))
            print(f'taste_noise={taste} input_noise={noise} missing={missing} {objective.id} done', flush=True)
    with (output/'sensitivity.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(output/'sensitivity.json', rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}, args.output)


if __name__ == '__main__':
    main()
