"""Run one experiment folder from several Colab runtimes at once.

Each runtime claims an (objective, replica) job by writing claim.json in the
job folder and refreshing it while the job runs. Other runtimes skip jobs
with a fresh claim, so no job is trained twice at the same time. Results are
identical to `python -m kiosk_adaptive.study` because every job seeds itself
and resumes from its own checkpoint; only the order and the machine change.

This file lives outside src/kiosk_adaptive on purpose: the experiment
manifest fingerprints every module there, and scheduling must not change it.

Order: replicas 0..first_reps-1 for every objective first, then the rest, so
that a deadline cut still leaves every objective with several replicas.
"""
import argparse
import json
import multiprocessing
import os
import socket
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from kiosk_adaptive import study
from kiosk_adaptive.design import lhs_objectives

STALE_SECONDS = 20 * 60   # a claim or job file untouched this long is free
HEARTBEAT_SECONDS = 120


class _Captured(Exception):
    pass


def prepare(config, output):
    """Let study.run validate, write/verify the manifest and build the
    catalog and audit exactly as it does, then stop before the first job."""
    captured = {}
    original = study.run_job

    def capture(objective, rep, cfg, settings, catalog, audit, out):
        captured.update(cfg=cfg, settings=settings, catalog=catalog, audit=audit)
        raise _Captured

    study.run_job = capture
    try:
        study.run(dict(config, workers=1), output)
    except _Captured:
        pass
    finally:
        study.run_job = original
    cfg = captured['cfg']
    objectives, _ = lhs_objectives(cfg['lhs_samples'], cfg['seed'])
    objectives = [replace(o, id=o.id + f'_p{p:g}', power=p) for p in cfg['powers'] for o in objectives]
    return captured, objectives


def ordered_jobs(objectives, replicas, first_reps):
    first = [(o, r) for r in range(min(first_reps, replicas)) for o in objectives]
    rest = [(o, r) for r in range(min(first_reps, replicas), replicas) for o in objectives]
    return first + rest


def _now():
    return time.time()


def _fresh(path):
    try:
        return _now() - path.stat().st_mtime < STALE_SECONDS
    except FileNotFoundError:
        return False


def busy_elsewhere(job, me):
    claim = job / 'claim.json'
    if claim.exists():
        try:
            owner = json.loads(claim.read_text(encoding='utf-8')).get('runtime')
        except (ValueError, OSError):
            owner = None
        if owner != me and _fresh(claim):
            return True
    # A runtime without claims (the plain study command) shows up as recently
    # written job files.
    for name in ('live.json', 'progress.json'):
        if _fresh(job / name) and not (claim.exists() and _fresh(claim)):
            return True
    return False


def write_claim(job, me):
    job.mkdir(parents=True, exist_ok=True)
    study.atomic_json(job / 'claim.json', dict(runtime=me, host=socket.gethostname(),
                                               updated_utc=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--first-reps', type=int, default=3)
    parser.add_argument('--runtime', default=None)
    args = parser.parse_args()
    me = args.runtime or f'{socket.gethostname()}-{uuid.uuid4().hex[:6]}'
    output = Path(args.output)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}
    ctx, objectives = prepare(config, output)
    cfg = ctx['cfg']
    jobs = ordered_jobs(objectives, cfg['replicas'], args.first_reps)
    path = lambda o, r: output / o.id / f'rep_{r}'
    print(f'runtime {me}: {len(jobs)} jobs, {args.workers} workers', flush=True)

    running = {}
    lock = threading.Lock()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(HEARTBEAT_SECONDS):
            with lock:
                jobs_now = list(running.values())
            for o, r in jobs_now:
                try:
                    write_claim(path(o, r), me)
                except OSError as exc:
                    print('heartbeat failed', exc, flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'))
    try:
        while True:
            with lock:
                mine = {(o.id, r) for o, r in running.values()}
            pending = []
            for o, r in jobs:
                job = path(o, r)
                if (job / 'summary.json').exists() or (o.id, r) in mine:
                    continue
                pending.append((o, r))
            if not pending and not running:
                break
            for o, r in pending:
                if len(running) >= args.workers:
                    break
                job = path(o, r)
                if busy_elsewhere(job, me):
                    continue
                write_claim(job, me)
                time.sleep(2)  # let a racing claim land, then re-check ownership
                try:
                    owner = json.loads((job / 'claim.json').read_text(encoding='utf-8')).get('runtime')
                except (ValueError, OSError):
                    owner = None
                if owner != me:
                    continue
                future = pool.submit(study.run_job, o, r, cfg, ctx['settings'], ctx['catalog'], ctx['audit'], output)
                with lock:
                    running[future] = (o, r)
                print(f'{datetime.now(timezone.utc):%H:%M:%S} start {o.id} rep_{r}', flush=True)
            if running:
                done, _ = wait(list(running), timeout=60, return_when=FIRST_COMPLETED)
                for future in done:
                    with lock:
                        o, r = running.pop(future)
                    try:
                        future.result()
                    except Exception as exc:  # leave the claim to go stale; another pass retries
                        print(f'FAILED {o.id} rep_{r}: {exc!r}', flush=True)
                        continue
                    try:
                        (path(o, r) / 'claim.json').unlink()
                    except FileNotFoundError:
                        pass
                    print(f'{datetime.now(timezone.utc):%H:%M:%S} done  {o.id} rep_{r}', flush=True)
                    aggregate(output, objectives, cfg)
            else:
                time.sleep(60)  # everything left is held by other runtimes
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
    aggregate(output, objectives, cfg)


def aggregate(output, objectives, cfg):
    """Rebuild the experiment-level files from every finished job on disk."""
    done = []
    for o in objectives:
        for r in range(cfg['replicas']):
            target = output / o.id / f'rep_{r}' / 'summary.json'
            if target.exists():
                done.append(json.loads(target.read_text(encoding='utf-8')))
    study.atomic_json(output / 'summary.json', done)
    study.atomic_json(output / 'objective_summary.json', study.combine_replicas(done, cfg['seed']))
    if done:
        study.export_csv(output, done)
    print(f'aggregated {len(done)} finished jobs', flush=True)


if __name__ == '__main__':
    main()
