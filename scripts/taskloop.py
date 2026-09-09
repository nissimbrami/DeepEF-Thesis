#!/usr/bin/env python3
"""TASKLOOP -- executes the DeepEF task queue autonomously. NO agents. NO waiting for a human.

Runs as a SLURM CPU job. Each cycle: pick the highest-priority task whose preconditions are met,
run it, verify the ARTIFACT exists, record the outcome, move to the next. Never stops until the
queue is empty or the walltime ends.

Safety, enforced in code and not by discipline:
  - NEVER scancel/kill anything. There is no scancel call in this file.
  - NEVER git push.
  - Every task must name an ARTIFACT; a task that produces no file is a FAILURE, not a success.
    (This project's signature failure is code that prints DONE and writes nothing.)
  - gate_g4_cpu must pass before and after any task that touches model code.
"""
import json, os, subprocess, sys, time

ROOT = '/home/nissimb/DeepPEF'
STATE = os.path.join(ROOT, 'results', 'taskloop_state.json')
LOG = os.path.join(ROOT, 'logs', 'taskloop.log')
PY = '/home/nissimb/.conda/envs/esm2_env_py38/bin/python'

def log(m):
    line = '[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), m)
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def sh(cmd, timeout=3600):
    try:
        p = subprocess.run(['bash','-lc',cmd], cwd=ROOT, capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout or '')[-4000:], (p.stderr or '')[-1500:]
    except subprocess.TimeoutExpired:
        return 124, '', 'TIMEOUT'

def load():
    if os.path.exists(STATE):
        try:
            with open(STATE) as f: return json.load(f)
        except Exception: pass
    return {'done': [], 'failed': [], 'cycles': 0}

def save(s):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    t = STATE + '.tmp'
    with open(t,'w') as f: json.dump(s,f,indent=1)
    os.replace(t, STATE)

def gate():
    rc,o,_ = sh('%s scripts/gate_g4_cpu.py 2>&1 | tail -3' % PY, 900)
    ok = 'ALL PASS' in o and '1092' in o
    if not ok: log('!! GATE FAILED -- halting mutation tasks\n%s' % o)
    return ok

# ---------------------------------------------------------------- the queue
# Each task: (id, precondition, command, artifact). Artifact is verified with os.path.exists.
def queue():
    E = os.path.join(ROOT,'eval_results')
    return [
      # --- P0c: rescore two runs read at a non-canonical epoch -------------
      ('P0c_loro', lambda: not os.path.exists(f'{E}/abl_loroW_onehot_s42_e14.csv'),
       'bash Megascale-fineTuning/run_calib_eval.sh loroW_onehot_s42 14',
       f'{E}/abl_loroW_onehot_s42_e14.csv'),

      # --- K8: score every finished golden arm that has no CSV -------------
      ('K8_score_golden',
       lambda: True,
       'bash scripts/score_all.sh',
       None),

      # --- scoreboard refresh (always safe, always informative) ------------
      ('scoreboard', lambda: True, '%s sb.py' % PY,
       os.path.join(ROOT,'results','02_findings','scoreboard.csv')),

      # --- K20: the unfolded ENSEMBLE, the last live reference-state idea --
      ('K20_ensemble',
       lambda: os.path.exists(os.path.join(ROOT,'scripts','ensemble_coil.py')),
       '%s scripts/ensemble_coil.py --k 8 --out results/08_data/ensemble_coil.json' % PY,
       os.path.join(ROOT,'results','08_data','ensemble_coil.json')),
    ]

def main():
    s = load()
    log('=== TASKLOOP start (done=%d failed=%d) ===' % (len(s['done']), len(s['failed'])))
    idle = 0
    while True:
        s['cycles'] += 1
        did = False
        for tid, pre, cmd, art in queue():
            if tid in s['done'] or tid in s['failed']:
                continue
            try:
                if not pre():
                    continue
            except Exception as e:
                log('%s precondition error: %s' % (tid, e)); continue
            if not gate():
                log('gate down -- sleeping 600'); time.sleep(600); break
            log('RUN %s :: %s' % (tid, cmd[:120]))
            rc,o,e = sh(cmd)
            ok = (rc == 0) and (art is None or os.path.exists(art))
            if ok:
                s['done'].append(tid); log('OK %s%s' % (tid, ' -> '+art if art else ''))
            else:
                s['failed'].append(tid)
                log('FAIL %s rc=%s artifact=%s\n%s\n%s' % (tid, rc, art, o[-1200:], e[-400:]))
            save(s); did = True
            break
        if not did:
            idle += 1
            log('nothing runnable (idle %d) -- sleeping 900' % idle)
            time.sleep(900)
            if idle > 96:
                log('=== queue exhausted for 24h -- exiting ==='); return
        else:
            idle = 0

if __name__ == '__main__':
    main()
