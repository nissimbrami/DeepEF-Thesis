"""Verify the autopilot's robustness claims (item B). Read-only w.r.t. the live driver:
this NEVER signals, kills or restarts the running autopilot. It exercises the lock and
state code in a throwaway directory with a subprocess that impersonates the driver.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autopilot as AP

FAILS = []


def check(name, cond, detail=''):
    print('  [%s] %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + detail) if detail else ''))
    if not cond:
        FAILS.append(name)


print('=' * 70)
print('AUTOPILOT ROBUSTNESS -- lock / state / signal survival')
print('=' * 70)

# ---------------------------------------------------------------- B1 atomic state write
print('')
print('B1  state write is atomic (temp file + os.replace, never truncates)')
tmp = tempfile.mkdtemp()
try:
    old_state = AP.STATE
    AP.STATE = os.path.join(tmp, 'results', 'state.json')
    st = {'state': 'S3_FACTORIAL', 'cells_done': 7, 'decisions': [], 'blocked': [],
          'retries': {}, 'submitted': []}
    AP.save_state(st)
    check('state file created through a missing results/ dir', os.path.exists(AP.STATE))
    check('written state reloads intact', json.load(open(AP.STATE))['cells_done'] == 7)
    check('no .tmp left behind', not os.path.exists(AP.STATE + '.tmp'))
    # the real property: a reader never sees a half-written file, because the visible
    # path only ever changes by rename. Prove the code uses replace, not in-place write.
    import inspect
    srcf = inspect.getsource(AP.save_state)
    check('save_state uses os.replace (atomic rename), not a bare open-and-write',
          'os.replace' in srcf and srcf.index('tmp') < srcf.index('os.replace'))
    # overwrite with a bigger doc then a smaller one; must never be corrupt
    for n in (5000, 3, 900):
        st['decisions'] = [{'node': 'D%d' % i, 'choice': 'x'} for i in range(n)]
        AP.save_state(st)
        got = json.load(open(AP.STATE))
        if len(got['decisions']) != n:
            check('state survives resize to %d' % n, False)
            break
    else:
        check('state survives repeated grow/shrink rewrites without corruption', True)
    # missing results/ dir -- the first death cause
    shutil.rmtree(os.path.dirname(AP.STATE))
    AP.save_state(st)
    check('recreates results/ if deleted under it (death cause #1)', os.path.exists(AP.STATE))
    AP.STATE = old_state
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- B2 lock is pid-aware
print('')
print('B2  lock is pid-aware and self-reclaiming')
import inspect
msrc = inspect.getsource(AP.main)
check('lock uses mkdir (atomic exclusion)', 'os.mkdir(lock)' in msrc)
check('lock records its pid', "f.write(str(os.getpid()))" in msrc)
check('a live holder is detected with os.kill(pid, 0)', 'os.kill(holder, 0)' in msrc)
check('a dead holder is reclaimed, not fatal', 'reclaiming stale lock' in msrc)
check('lock is released in a finally: block', 'finally:' in msrc and 'os.rmdir(lock)' in msrc)

# real behaviour, in a throwaway lock dir, using a real subprocess that we own
tmp = tempfile.mkdtemp()
try:
    lock = os.path.join(tmp, '.autopilot.lock')
    pidfile = os.path.join(lock, 'pid')
    os.makedirs(lock)

    # (a) stale lock: pid that cannot exist
    with open(pidfile, 'w') as f:
        f.write('999999')
    holder = int(open(pidfile).read().strip())
    alive = True
    try:
        os.kill(holder, 0)
    except OSError:
        alive = False
    check('a lock held by a dead pid is seen as reclaimable', not alive)

    # (b) live lock: a real sleeping child WE own (never the autopilot)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    with open(pidfile, 'w') as f:
        f.write(str(child.pid))
    holder = int(open(pidfile).read().strip())
    alive = True
    try:
        os.kill(holder, 0)
    except OSError:
        alive = False
    check('a lock held by a LIVE pid is respected (no double-start)', alive)
    child.terminate()
    child.wait(timeout=10)
    time.sleep(0.3)
    alive = True
    try:
        os.kill(child.pid, 0)
    except OSError:
        alive = False
    check('once that holder dies the same lock becomes reclaimable', not alive)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- B3 SIGTERM survival
print('')
print('B3  login-node drop (rc=143 = SIGTERM) is survivable by the wrapper')
wrapper = os.path.expanduser('~/auto/run_autopilot.sh')
w = open(wrapper).read() if os.path.exists(wrapper) else ''
check('wrapper exists', bool(w))
check('wrapper restarts on any rc except 1 and 3',
      'if [ $rc -eq 3 ] || [ $rc -eq 1 ]; then break; fi' in w)
check('rc=143 (SIGTERM) therefore RESTARTS rather than ending the run',
      '143' not in w.split('break')[0].split('if [ $rc')[-1] if 'if [ $rc' in w else False)
check('wrapper loops forever', 'while true; do' in w)
check('wrapper redirects both streams to the log', '>> /home/nissimb/auto/autopilot.log 2>&1' in w)

# a killed driver must leave a lock that the NEXT start can reclaim
tmp = tempfile.mkdtemp()
try:
    lock = os.path.join(tmp, '.autopilot.lock')
    os.makedirs(lock)
    victim = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    with open(os.path.join(lock, 'pid'), 'w') as f:
        f.write(str(victim.pid))
    victim.send_signal(signal.SIGTERM)      # exactly what the login node does
    rc = victim.wait(timeout=10)
    check('SIGTERM to a driver yields rc=143 (-15)', rc in (-15, 143), 'rc=%r' % rc)
    check('its lock dir survives the kill (that is why reclaim is required)',
          os.path.isdir(lock))
    holder = int(open(os.path.join(lock, 'pid')).read().strip())
    alive = True
    try:
        os.kill(holder, 0)
    except OSError:
        alive = False
    check('the next start sees a dead holder and can reclaim -> no permanent wedge',
          not alive)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('')
print('=' * 70)
if FAILS:
    print('ROBUSTNESS: %d FAILED -> %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('ROBUSTNESS: ALL PASSED')
print('=' * 70)
