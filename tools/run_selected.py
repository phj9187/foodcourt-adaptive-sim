"""Run only selected objectives of an experiment folder locally (deadline quick tests)."""
import argparse, sys, yaml
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cooperative_run import prepare, _cpu_only_seeding
from kiosk_adaptive import study

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--only', required=True); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    cfg_in = yaml.safe_load(open(a.config, encoding='utf-8'))
    cap, objectives = prepare(cfg_in, a.output)
    want = [w.strip() for w in a.only.split(',')]
    sel = [o for o in objectives if o.id.split('_p')[0] in want or o.id in want]
    jobs = [(o, r) for o in sel for r in range(cap['cfg']['replicas'])]
    print('jobs', [(o.id, r) for o, r in jobs], flush=True)
    with ProcessPoolExecutor(a.workers, initializer=_cpu_only_seeding) as pool:
        fs = [pool.submit(study.run_job, o, r, cap['cfg'], cap['settings'], cap['catalog'], cap['audit'], Path(a.output)) for o, r in jobs]
        for f in fs: s = f.result(); print('done', s['objective']['id'], flush=True)

if __name__ == '__main__':
    main()
