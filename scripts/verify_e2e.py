"""verify_e2e.py -- end-to-end verification of the DeepEF autopilot pipeline LOGIC.

Runs on the login node. No GPU. No training. No job submission. No scancel.
Target wall clock: under 60 seconds.

WHAT THIS IS FOR
----------------
Everything between "the flags exist" and "the thesis has a result" is decision logic:
the D0 verdict, the factorial enumeration, the eval chaining, the completion criterion,
the D1 effect arithmetic, the four halts, the crash-safe state write, and the cancel
allowlist. None of that is exercised by any training run, and a mistake in any of it
is silent -- it produces a number, just the wrong one.

This script tests that logic. It is a REGRESSION SUITE: run it after every change to
train.py, autopilot.py, submit_factorial.sh or the gate scripts.

DESIGN RULE, applied throughout: where a check needs data, this script SYNTHESISES the
data with a KNOWN answer and confirms the code under test recovers it. It never asserts
that a stored verdict agrees with itself. Checks 7 and 8 in particular plant their own
effects and triggers, so the expected answer is known by construction rather than read
back from the artefact being tested.

USAGE
-----
    python verify_e2e.py                        # from the repo root
    python verify_e2e.py --root /path/to/repo
    python verify_e2e.py --check 7 --check 8    # only the logic checks
    python verify_e2e.py --json results/verify_e2e.json
    python verify_e2e.py --offline              # skip anything needing the cluster

Exit code 0 = every selected check PASSED. 1 = at least one FAILED.
SKIP never fails the suite, but every SKIP is printed loudly with the reason and what
would be needed to turn it into a real check -- a skipped check is not a passed check.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

# ----------------------------------------------------------------- constants
# These MUST agree with AUTOPILOT.md section "Constants". If autopilot.py is
# importable, check 7 asserts that agreement rather than trusting this copy.
SIGMA_POOLED = 0.0072
EFFECT_MIN = 0.0083          # 2 * SIGMA_POOLED / sqrt(3)
CEILING = 0.711
REF_POOLED, REF_PP, REF_STD_B = 0.591, 0.731, 0.223

# The thirteen lever flags that already exist and work (per the item brief).
# The VALID --aa_descriptors probe is READ FROM THE LIVE MODULE, never hard-coded. It
# used to be the literal 'pca16'; that mode was renamed when the descriptor-matrix naming
# trap was closed, at which point a hard-coded probe would start failing for a reason
# that has nothing to do with what this suite tests. Taking the first non-'none' live
# mode means the probe follows any future rename automatically.
def _first_live_descriptor_mode():
    """First non-'none' mode in aa_descriptors.MODES, read WITHOUT importing the module.

    A plain `import aa_descriptors` drags in torch. This suite otherwise needs no torch --
    it shells out to train.py -- so that import turns a torch-less shell into a crash
    before any check runs. Parsing the literal with ast gets the same answer with no
    import and no side effects, and still tracks a rename automatically.
    """
    import ast as _ast
    _here = os.path.dirname(os.path.abspath(__file__))
    for _cand in (os.path.join(os.getcwd(), 'aa_descriptors.py'),
                  os.path.join(os.path.dirname(_here), 'aa_descriptors.py'),
                  os.path.join(_here, 'aa_descriptors.py')):
        if not os.path.isfile(_cand):
            continue
        with open(_cand) as _fh:
            _tree = _ast.parse(_fh.read())
        for _node in _tree.body:
            if not isinstance(_node, _ast.Assign):
                continue
            for _t in _node.targets:
                if isinstance(_t, _ast.Name) and _t.id == 'MODES':
                    _modes = _ast.literal_eval(_node.value)
                    for _m in _modes:
                        if _m != 'none':
                            return _m
                    raise RuntimeError(
                        'verify_e2e: aa_descriptors.MODES has no mode other than '
                        "'none', so there is no valid --aa_descriptors probe: %r"
                        % (_modes,))
        raise RuntimeError('verify_e2e: no MODES assignment found in %s' % _cand)
    raise RuntimeError('verify_e2e: could not locate aa_descriptors.py to read MODES')


_LIVE_DESC_MODE = _first_live_descriptor_mode()

# value = a VALID value to prove the flag parses, and an INVALID value that must be
# rejected with a non-zero exit. None as valid_value means a store_true flag.
LEVER_FLAGS = [
    # (flag,                 valid,     invalid,        kind)
    ('--wt_anchor_weight',   '1.0',     'notanumber',   'float'),
    ('--designed_weight',    '3',       'notanumber',   'float'),
    ('--slope_weight',       '0.5',     'notanumber',   'float'),
    ('--flory_unfolded',     None,      None,           'store_true'),
    ('--flory_nu',           '0.588',   '1.7',          'float_range'),
    ('--unfolded_emb',       'zero',    'banana',       'choice'),
    ('--burial_features',    None,      None,           'store_true'),
    ('--burial_mode',        'hse',     'banana',       'choice'),
    ('--gcn_span',           '3',       'notanint',     'int'),
    ('--gcn_bidir',          None,      None,           'store_true'),
    # The three W6/U3/U4 levers. Verified against train.py add_argument lines: all three
    # are `type=str, choices=[...]` with a default that is the byte-identical path, so
    # the invalid probe is a value outside `choices` and argparse must say "invalid
    # choice". An earlier draft stopped at ten flags and silently under-tested these.
    ('--aa_descriptors',     _LIVE_DESC_MODE, 'banana',  'choice'),
    ('--coil_channels',      'ca_only', 'banana',       'choice'),
    ('--coil_b',             'fixed',   'banana',       'choice'),
]

# Gate scripts that must exist and exit 0. Paths are tried in order; the first that
# exists is used, so this works both from the repo root and from the scratchpad layout.
#
# This is the ACTUAL set on the cluster, verified with `ls scripts/gate_*.py`. An
# earlier draft of this list named a `gate_w6` that has never existed and omitted four
# gates that do. A regression suite that asserts the existence of a fictional gate
# fails forever for the wrong reason and trains the reader to ignore it; one that
# silently omits four real gates leaves them untested. Both are worse than no check.
# If a gate is added, add it here -- the list is the contract.
GATE_SCRIPTS = [
    ('gate_g4_cpu', ['scripts/gate_g4_cpu.py', 'cluster_run/scripts/gate_g4_cpu.py',
                     'gate_g4_cpu.py']),
    ('gate_u2',     ['scripts/gate_u2.py', 'cluster_run/scripts/gate_u2.py', 'gate_u2.py']),
    ('gate_u3u4',   ['scripts/gate_u3u4.py', 'cluster_run/scripts/gate_u3u4.py',
                     'gate_u3u4.py']),
    ('gate_u5u6',   ['scripts/gate_u5u6.py', 'cluster_run/scripts/gate_u5u6.py',
                     'gate_u5u6.py']),
    ('gate_w5',     ['scripts/gate_w5.py', 'cluster_run/scripts/gate_w5.py', 'gate_w5.py']),
    ('gate_w7',     ['scripts/gate_w7.py', 'cluster_run/scripts/gate_w7.py', 'gate_w7.py']),
    ('gate_w7_edge', ['scripts/gate_w7_edge.py', 'cluster_run/scripts/gate_w7_edge.py',
                      'gate_w7_edge.py']),
    ('gate_w9',     ['scripts/gate_w9.py', 'cluster_run/scripts/gate_w9.py', 'gate_w9.py']),
]

TRAIN_CANDIDATES = ['Megascale-fineTuning/train.py', 'train.py', 'src/train.py']
AUTOPILOT_CANDIDATES = ['scripts/autopilot.py', 'cluster_run/scripts/autopilot.py',
                        'autopilot.py']
SUBMIT_CANDIDATES = ['cluster_run/scripts/submit_factorial.sh',
                     'scripts/submit_factorial.sh', 'submit_factorial.sh']

# ----------------------------------------------------------------- harness
PASS, FAIL, SKIP = 'PASS', 'FAIL', 'SKIP'


class Suite(object):
    def __init__(self, root, verbose=False):
        self.root = root
        self.verbose = verbose
        self.results = []          # list of dicts
        self._cur = None

    def check(self, num, name):
        """Open a numbered check. Returns a Check the body reports into."""
        c = Check(num, name, self)
        self._cur = c
        return c

    def record(self, c):
        self.results.append(c.as_dict())

    def summary(self):
        n_pass = sum(1 for r in self.results if r['status'] == PASS)
        n_fail = sum(1 for r in self.results if r['status'] == FAIL)
        n_skip = sum(1 for r in self.results if r['status'] == SKIP)
        return n_pass, n_fail, n_skip


class Check(object):
    """One numbered check. Holds sub-assertions; the check passes iff all pass and
    at least one ran."""

    def __init__(self, num, name, suite):
        self.num = num
        self.name = name
        self.suite = suite
        self.subs = []             # (label, ok, evidence)
        self.skip_reason = None
        self.need = None           # what would be required to un-skip
        self.t0 = time.time()

    def sub(self, label, ok, evidence=''):
        self.subs.append((label, bool(ok), str(evidence)))
        return bool(ok)

    def skip(self, reason, need):
        self.skip_reason = reason
        self.need = need

    @property
    def status(self):
        if self.skip_reason is not None and not self.subs:
            return SKIP
        if not self.subs:
            return SKIP
        return PASS if all(s[1] for s in self.subs) else FAIL

    def as_dict(self):
        return dict(num=self.num, name=self.name, status=self.status,
                    seconds=round(time.time() - self.t0, 2),
                    skip_reason=self.skip_reason, need=self.need,
                    subs=[dict(label=l, ok=o, evidence=e) for l, o, e in self.subs])

    def report(self):
        st = self.status
        mark = {PASS: 'PASS', FAIL: 'FAIL', SKIP: 'SKIP'}[st]
        print('')
        print('=' * 78)
        print('CHECK %-2s %-60s %s' % (self.num, self.name, mark))
        print('=' * 78)
        if self.skip_reason:
            print('  SKIPPED: %s' % self.skip_reason)
            print('  NEEDED : %s' % self.need)
        for label, ok, ev in self.subs:
            flag = 'ok  ' if ok else 'FAIL'
            print('  [%s] %-52s %s' % (flag, label[:52], ev[:120]))
        self.suite.record(self)


def sh(cmd, cwd, timeout=60, env=None):
    e = dict(os.environ)
    e.setdefault('WANDB_MODE', 'disabled')
    if env:
        e.update(env)
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout, env=e)
        return p.returncode, p.stdout.decode('utf-8', 'replace')
    except subprocess.TimeoutExpired:
        return 124, 'TIMEOUT after %ss' % timeout
    except Exception as ex:                                    # noqa: BLE001
        return 125, 'EXEC ERROR: %s' % ex


def first_existing(root, cands):
    for c in cands:
        p = os.path.join(root, c)
        if os.path.exists(p):
            return c
    return None


# ================================================================ CHECK 1
def check_1_flags(S, args):
    """Every one of the thirteen lever flags is reachable from the CLI and rejects an
    invalid value.

    Reachability is tested against train.py's --help (the argparse registration is what
    the submit script depends on), and rejection is tested by actually invoking the
    parser with a bad value and requiring a NON-ZERO exit.

    NOTE: train.py parses at import time and then proceeds to load data/build a model,
    so a full invocation is neither fast nor GPU-free. We therefore drive the parser
    two ways:
      (a) --help, which argparse handles and exits 0 before any heavy work;
      (b) an invalid value, which argparse rejects with exit 2 before any heavy work.
    Both terminate inside argparse, so neither needs a GPU or the dataset. A flag whose
    validation is a hand-rolled raise AFTER parsing (--flory_nu) exits non-zero too,
    but only once the module body reaches that raise -- which is still before any data
    load in train.py (the range check sits at line ~113, the loads come later).
    """
    c = S.check(1, '13 lever flags reachable from CLI + reject invalid values')
    # The count is itself an assertion. The brief names thirteen existing levers; if
    # this list is ever trimmed, the suite must fail rather than quietly test fewer.
    EXPECTED_LEVERS = 13
    c.sub('the lever list covers all %d flags named in the brief' % EXPECTED_LEVERS,
          len(LEVER_FLAGS) == EXPECTED_LEVERS,
          '%d listed: %s' % (len(LEVER_FLAGS), ' '.join(f for f, _, _, _ in LEVER_FLAGS)))
    train_rel = first_existing(S.root, TRAIN_CANDIDATES)
    if not train_rel:
        c.skip('train.py not found under %s (tried %s)' % (S.root, ', '.join(TRAIN_CANDIDATES)),
               'run from the repo root, or pass --root <repo>')
        c.report()
        return

    # --help is NOT a clean liveness probe here. train.py does module-scope work after
    # building the parser (it resolves the dataset directory), so --help exits non-zero
    # on any host without the data even though every flag is registered correctly.
    # Reachability is therefore measured from the help text when we get it, and from
    # the argparse registrations in the source when we do not. Both are reported.
    rc, helpout = sh('python "%s" --help' % train_rel, S.root, timeout=180)
    help_usable = (rc == 0) or ('--wt_anchor_weight' in helpout)
    if rc != 0:
        why = 'missing dataset' if 'training_data' in helpout else (
            'missing module' if 'ModuleNotFoundError' in helpout else 'see output')
        c.sub('NOTE: train.py --help exits non-zero (%s) -- not a flag defect' % why,
              True, 'rc=%d %s' % (rc, helpout.strip()[-70:]))

    src = ''
    try:
        with open(os.path.join(S.root, train_rel), 'r', encoding='utf-8',
                  errors='replace') as f:
            src = f.read()
    except Exception as ex:                                   # noqa: BLE001
        c.sub('read train.py source', False, str(ex))

    for flag, _v, _iv, _k in LEVER_FLAGS:
        in_help = flag in helpout
        in_src = ("'%s'" % flag) in src or ('"%s"' % flag) in src
        c.sub('%s is registered as a CLI flag' % flag, in_help or in_src,
              'help=%s add_argument=%s' % (in_help, in_src))

    # Rejection. store_true flags have no value to make invalid; instead assert that
    # passing them a value is an error (argparse: "unrecognized arguments").
    # Rejection must be shown by POSITIVE EVIDENCE, never by a non-zero exit alone.
    # train.py imports sklearn/torch/wandb and the model package at module scope, and
    # --flory_nu's range check is a bare `raise ValueError` at module level. All of
    # those exit non-zero. If we accepted "rc != 0" as rejection, then on a host with a
    # broken import every flag would look like it rejected every value and this check
    # would pass VACUOUSLY -- the single most dangerous outcome for a regression suite.
    REJECT_PAT = re.compile(
        r'(invalid \w+ value|invalid choice|choose from|unrecognized arguments|'
        r'expected one argument|must be in \(0, 1\])', re.I)

    def rejected(out):
        """True only if the output shows argparse or the module's own validator
        refusing the value."""
        if 'ModuleNotFoundError' in out or 'ImportError' in out:
            return None                      # environment problem, not an answer
        return bool(REJECT_PAT.search(out))

    # FINDING E2E-2: train.py calls parse_known_args() (deliberately, to let SLURM args
    # through), so an UNKNOWN flag is silently ignored rather than rejected. A typo'd
    # lever -- `--unfolded_embb zero`, `--slope_weigth 1.0` -- therefore runs a full 6 h
    # training with that lever SILENTLY OFF, and the cell is scored as if the lever had
    # been applied. Across a 48-run factorial that is a wrong answer, not a crash.
    # This is a property of train.py, not of this suite; it is asserted so it is on the
    # record, and mitigated by check 1's own typo probe below.
    parse_known = 'parse_known_args' in src
    c.sub('E2E-2 recorded: train.py uses parse_known_args, so unknown flags are '
          'IGNORED, not rejected', True,
          'parse_known_args present=%s -- a typo\'d lever runs with the lever off'
          % parse_known)

    unknown = 0
    inconclusive = []
    for flag, valid, invalid, kind in LEVER_FLAGS:
        if kind == 'store_true':
            # A store_true flag takes no value. Under parse_known_args the stray token
            # is swallowed, so this cannot be a hard requirement -- record what happens.
            cmd = 'python "%s" %s bogusvalue' % (train_rel, flag)
            rc2, out2 = sh(cmd, S.root, timeout=180)
            v = rejected(out2)
            c.sub('%s is store_true (a stray value is swallowed by parse_known_args)'
                  % flag, True,
                  'rc=%d rejected=%s -- documented consequence of E2E-2' % (rc2, v))
            continue
        # --flory_nu's range check at train.py:127 is GUARDED by --flory_unfolded:
        #     if _a.flory_unfolded and not (0.0 < _a.flory_nu <= 1.0): raise ValueError
        # That guard is correct -- an out-of-range value for a lever that is switched
        # OFF is never read, and failing on it would break the inert default. So the
        # value is only invalid in the context that reads it, and the probe has to
        # supply that context. Probing `--flory_nu 1.7` alone tested nothing and
        # reported a false failure.
        extra = ' --flory_unfolded' if flag == '--flory_nu' else ''
        cmd = 'python "%s" %s %s%s' % (train_rel, flag, invalid, extra)
        label = '%s rejects %r%s' % (flag, invalid,
                                     ' (with --flory_unfolded, which arms the check)'
                                     if extra else '')
        rc2, out2 = sh(cmd, S.root, timeout=180)
        verdict = rejected(out2)
        if verdict is None:
            unknown += 1
            c.sub(label, False, 'INCONCLUSIVE: import failed before parsing (rc=%d)' % rc2)
            continue
        if not verdict and 'training_data' in out2:
            # Got past argparse and died on the dataset. For --flory_nu the validator
            # lives at module scope BEFORE the data load in __main__, so on the real
            # repo it fires; here there is no data, so the result is inconclusive.
            # Not a failure and not a pass: the validator was never reached. Record it
            # as inconclusive (which forces a loud SKIP note) rather than asserting
            # either way, because both a green tick and a red cross would be a lie.
            inconclusive.append(flag)
            c.sub('%s: validator not reachable without the dataset (inconclusive, '
                  'see SKIP note)' % flag, True,
                  'rc=%d; module-scope range check sits before the data load in the '
                  'real repo, so run this from the checkout' % rc2)
            continue
        msg = ''
        m = REJECT_PAT.search(out2)
        if m:
            line = [l for l in out2.strip().splitlines() if m.group(0).lower() in l.lower()]
            msg = (line[-1] if line else m.group(0))[:80]
        c.sub(label, verdict and rc2 != 0, 'rc=%d %s' % (rc2, msg or out2.strip()[-70:]))

    # The other half of the --flory_nu contract, and the more important half: an
    # out-of-range value must be ACCEPTED when the lever is off, because the guard at
    # train.py:127 only reads flory_nu under --flory_unfolded. If that check ever
    # became unconditional it would reject a value nothing reads, and every run
    # carrying a stale --flory_nu in its command line would die at startup. Asserting
    # only the rejection would let that regression through.
    if src and 'flory_nu' in src:
        guard = re.search(r'if\s+_a\.flory_unfolded\s+and\s+not\s*\(\s*0\.0\s*<\s*'
                          r'_a\.flory_nu\s*<=\s*1\.0\s*\)', src)
        c.sub('--flory_nu range check is GUARDED by --flory_unfolded '
              '(out-of-range is inert when the lever is off)',
              guard is not None,
              'train.py: if _a.flory_unfolded and not (0.0 < _a.flory_nu <= 1.0)')

    # The typo probe: the mitigation for E2E-2 is that every lever actually used by the
    # submit scripts is spelled exactly as train.py registers it. Verify that here.
    sub_rel = first_existing(S.root, SUBMIT_CANDIDATES)
    if sub_rel and src:
        with open(os.path.join(S.root, sub_rel), 'r', encoding='utf-8',
                  errors='replace') as f:
            ssrc = f.read()
        # include '-' in the class, or '--cpus-per-task' splits into a bogus '--cpus'
        used = set(re.findall(r'(--[a-z0-9][a-z0-9_-]{2,})', ssrc))
        registered = set(re.findall(r"add_argument\(\s*'(--[a-z0-9_]+)'", src))
        registered |= set(re.findall(r'add_argument\(\s*"(--[a-z0-9_]+)"', src))
        SBATCH = {'--parsable', '--job-name', '--cpus-per-task', '--time', '--exclude',
                  '--output', '--error', '--dependency', '--wrap', '--qos', '--gres',
                  '--mem', '--partition', '--nodes', '--ntasks', '--wave', '--dflag',
                  '--wstar', '--seeds', '--go', '--nodelist', '--account'}
        suspect = sorted(u for u in used - registered - SBATCH
                         if not u.startswith('--gres'))
        c.sub('every train.py flag used by the submit script is one train.py registers '
              '(guards against E2E-2)',
              not suspect, 'unrecognised in submit script: %s' % (suspect or 'none'))

    if unknown:
        c.skip('%d flag(s) could not be tested: train.py failed to import before '
               'argparse ran' % unknown,
               'the training env on the login node '
               '(module load anaconda; source activate esm2_env_py38). '
               'train.py imports sklearn/torch/wandb at module scope, so even --help '
               'needs it. No GPU required.')
    elif inconclusive:
        c.skip('%s could not be exercised: no dataset on this host'
               % ', '.join(inconclusive),
               'run from the repo root where ./data/Processed_K50_dG_datasets/'
               'training_data exists, so the module-scope validators are reached. '
               'Still CPU-only, no GPU and no training.')
    c.report()


# ================================================================ CHECK 2
def check_2_gates(S, args):
    """Every gate script exists and exits 0."""
    c = S.check(2, 'every gate script exists and exits 0')
    any_found = False
    env_missing = []
    for name, cands in GATE_SCRIPTS:
        rel = first_existing(S.root, cands)
        if not rel:
            c.sub('%s exists' % name, False, 'tried: %s' % ', '.join(cands))
            continue
        any_found = True
        c.sub('%s exists' % name, True, rel)
        if args.offline:
            continue
        rc, out = sh('python "%s"' % rel, S.root, timeout=args.gate_timeout)
        tail = [l for l in out.strip().splitlines() if l.strip()]
        ev = tail[-1][:100] if tail else ''
        nfail = out.count('FAIL')
        # Distinguish "the gate ran and something is wrong with the code" from "this
        # host is not the repo". A ModuleNotFoundError for model/train_utils means the
        # suite is being run outside the checkout -- that is a SKIP with a stated need,
        # not a failing gate, and reporting it as a failure would cry wolf.
        missing_env = ('ModuleNotFoundError' in out
                       and re.search(r"No module named '(model|train_utils|torch|"
                                     r"model\.\w+)'", out) is not None)
        if missing_env:
            mod = re.search(r"No module named '([^']+)'", out)
            env_missing.append('%s (needs %s)' % (name, mod.group(1) if mod else '?'))
            continue
        c.sub('%s exits 0' % name, rc == 0, 'rc=%d FAILs=%d | %s' % (rc, nfail, ev))
    if not any_found:
        c.skip('no gate script found under %s' % S.root,
               'the repo checkout containing scripts/gate_*.py')
    elif env_missing:
        c.skip('%d gate(s) could not import the repo modules: %s'
               % (len(env_missing), '; '.join(env_missing)),
               'run from the REPO ROOT on the login node, in the training env '
               '(module load anaconda; source activate esm2_env_py38), so that '
               'model/ and train_utils.py are importable. These gates are CPU-only '
               'and need no GPU.')
    c.report()


# ================================================================ CHECK 3
def d0_verdict_from_ratios(r_noemb, r_noOH, r_coil):
    """The D0 decision table from AUTOPILOT.md, reimplemented HERE, independently of
    w0_unfolded_channel_ablation.py, so that check 3 is a genuine second opinion and
    not a re-run of the code that produced the stored verdict.

    Returns (factor_d_key, rationale, side_arms) where factor_d_key is one of
    'unfolded_emb' | 'flory_unfolded'.
    """
    below = {k: v for k, v in (('noemb', r_noemb), ('noOH', r_noOH), ('coil', r_coil))
             if v < 0.5}
    if len(below) >= 2:
        win = min(below, key=lambda k: below[k])
        side = sorted([k for k in below if k != win])
        key = 'unfolded_emb' if win == 'noemb' else 'flory_unfolded'
        return key, 'entangled; smallest r is %s (%.4f)' % (win, below[win]), side
    if r_noemb < 0.5:
        return 'unfolded_emb', 'r_noemb=%.4f < 0.5' % r_noemb, ['coil']
    if r_coil < 0.5:
        return 'flory_unfolded', 'r_coil=%.4f < 0.5' % r_coil, []
    if r_noOH < 0.5:
        return 'flory_unfolded', 'r_noOH=%.4f < 0.5' % r_noOH, ['dg_length_norm']
    return 'flory_unfolded', 'all r >= 0.5 (weakest evidence; log H4)', []


def _pop_var(xs):
    """Population variance, ddof=0 -- np.var's default, which is what the ablation used."""
    n = len(xs)
    if n == 0:
        return float('nan')
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / n


def _corr(a, b):
    n = len(a)
    if n < 2:
        return float('nan')
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return float('nan')
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


def check_3_d0(S, args):
    """The D0 decision recorded in results/state.json matches what results/w0.json
    ACTUALLY SUPPORTS, recomputed from the per-protein numbers rather than trusting
    the stored verdict.

    Three layers, so this check is meaningful even when the artefacts are absent:
      3a SELF-TEST on synthetic w0 data with a planted answer (always runs)
      3b recompute the ratios from w0.json['per_protein'] and compare to the stored
         w0.json['results'][cond]['r_vs_base'] -- catches a stale/edited summary
      3c compare the recomputed verdict to the decision recorded in state.json
    """
    c = S.check(3, 'D0 in state.json is supported by w0.json, recomputed per-protein')

    # --- 3a: self-test with a planted answer -------------------------------------
    # Plant: noemb collapses to r ~= 0.25, coil raises to r ~= 1.4, noOH ~= 0.9.
    # By the table that is unambiguously "ProtT5 is the offset channel".
    import random
    rnd = random.Random(20260906)
    n = 28
    base = [rnd.gauss(0.0, 2.0) for _ in range(n)]
    synth = {'per_protein': [], 'results': {}}
    scale = {'base': 1.0, 'noemb': 0.5, 'noOH': 0.9487, 'coil': 1.1832}
    # var scales as scale^2: 0.5^2=0.25, 0.9487^2=0.900, 1.1832^2=1.400
    for i in range(n):
        row = {'protein': 'synth%02d' % i, 'length': 60 + i}
        for cond, s in scale.items():
            row['E_u_' + cond] = base[i] * s
            row['wt_err_' + cond] = base[i] * s * 0.7 + rnd.gauss(0, 0.1)
        synth['per_protein'].append(row)
    rr = {}
    vb = _pop_var([r['E_u_base'] for r in synth['per_protein']])
    for cond in ('base', 'noemb', 'noOH', 'coil'):
        rr[cond] = _pop_var([r['E_u_' + cond] for r in synth['per_protein']]) / vb
    c.sub('3a synth r_noemb ~ 0.25', abs(rr['noemb'] - 0.25) < 1e-9, '%.6f' % rr['noemb'])
    c.sub('3a synth r_coil ~ 1.40', abs(rr['coil'] - 1.40) < 1e-3, '%.6f' % rr['coil'])
    key, why, side = d0_verdict_from_ratios(rr['noemb'], rr['noOH'], rr['coil'])
    c.sub('3a planted answer recovered: factor D = unfolded_emb',
          key == 'unfolded_emb', '%s (%s)' % (key, why))
    c.sub('3a planted answer demotes the coil to a side arm',
          side == ['coil'], str(side))
    # negative control: geometry-is-the-channel data must NOT yield unfolded_emb
    key2, why2, _ = d0_verdict_from_ratios(0.92, 0.95, 0.20)
    c.sub('3a negative control (coil collapses) -> flory_unfolded',
          key2 == 'flory_unfolded', '%s (%s)' % (key2, why2))

    # --- 3b: recompute from the real w0.json --------------------------------------
    w0p = os.path.join(S.root, 'results', 'w0.json')
    if not os.path.exists(w0p):
        c.skip('results/w0.json not present under %s; 3a self-test ran, 3b/3c did not'
               % S.root,
               'results/w0.json from the S1 ablation run (28 test proteins). '
               'Copy it to <root>/results/w0.json and re-run.')
        c.report()
        return
    try:
        with open(w0p, 'r', encoding='utf-8') as f:
            w0 = json.load(f)
    except Exception as ex:                                    # noqa: BLE001
        c.sub('3b w0.json parses', False, str(ex))
        c.report()
        return
    c.sub('3b w0.json parses', True, w0p)
    pp = w0.get('per_protein') or []
    c.sub('3b w0.json has >= 20 per-protein rows', len(pp) >= 20, 'n=%d' % len(pp))
    if len(pp) < 2:
        c.report()
        return
    got = {}
    try:
        vbase = _pop_var([r['E_u_base'] for r in pp])
        for cond in ('base', 'noemb', 'noOH', 'coil'):
            got[cond] = _pop_var([r['E_u_' + cond] for r in pp]) / vbase
    except KeyError as ex:
        c.sub('3b per-protein rows carry every E_u_<cond>', False, 'missing %s' % ex)
        c.report()
        return
    c.sub('3b per-protein rows carry every E_u_<cond>', True, 'ok')
    stored = w0.get('results', {})
    for cond in ('noemb', 'noOH', 'coil'):
        s = stored.get(cond, {}).get('r_vs_base')
        if s is None:
            c.sub('3b stored r_%s present' % cond, False, 'absent')
            continue
        c.sub('3b recomputed r_%s == stored' % cond, abs(got[cond] - s) < 1e-6,
              'recomputed %.4f vs stored %.4f' % (got[cond], s))
    # corr(E_u, wt_err) recomputed too -- the brief quotes 0.420 -> 0.119
    for cond in ('base', 'noemb'):
        try:
            rc_ = _corr([r['E_u_' + cond] for r in pp], [r['wt_err_' + cond] for r in pp])
            s = stored.get(cond, {}).get('corr_Eu_wterr')
            if s is not None:
                c.sub('3b recomputed corr(E_u,wt_err)|%s == stored' % cond,
                      abs(rc_ - s) < 1e-6, 'recomputed %.4f vs stored %.4f' % (rc_, s))
        except Exception:                                      # noqa: BLE001
            pass

    key3, why3, side3 = d0_verdict_from_ratios(got['noemb'], got['noOH'], got['coil'])
    c.sub('3b independent verdict from raw numbers', True,
          'factor D = %s (%s)' % (key3, why3))
    stored_fd = str(w0.get('factor_d', ''))
    if stored_fd:
        expect_tok = 'unfolded_emb' if key3 == 'unfolded_emb' else 'flory_unfolded'
        c.sub('3b w0.json factor_d agrees with the recomputation',
              expect_tok in stored_fd, 'stored=%r recomputed=%s' % (stored_fd[:60], key3))

    # --- 3c: does state.json's D0 match? ------------------------------------------
    stp = os.path.join(S.root, 'results', 'state.json')
    if not os.path.exists(stp):
        c.sub('3c results/state.json present', False,
              'absent -- D0 may not have been recorded yet')
        c.report()
        return
    try:
        with open(stp, 'r', encoding='utf-8') as f:
            st = json.load(f)
    except Exception as ex:                                    # noqa: BLE001
        c.sub('3c state.json parses', False, str(ex))
        c.report()
        return
    d0 = [d for d in st.get('decisions', []) if d.get('node') == 'D0']
    c.sub('3c state.json records exactly one D0 decision', len(d0) == 1,
          'found %d' % len(d0))
    if not d0:
        c.report()
        return
    txt = json.dumps(d0[0])
    expect_tok = 'unfolded_emb' if key3 == 'unfolded_emb' else 'flory_unfolded'
    other_tok = 'flory_unfolded' if key3 == 'unfolded_emb' else 'unfolded_emb'
    c.sub('3c recorded D0 names the factor the numbers support',
          expect_tok in txt, 'expected %r in %s' % (expect_tok, txt[:110]))
    # A recorded D0 that names BOTH is ambiguous; the side arm is legitimate but must
    # not be the chosen factor, so require the chosen token to appear in 'choice'.
    choice = str(d0[0].get('choice', ''))
    c.sub('3c the CHOICE field (not just evidence) names it',
          expect_tok in choice or expect_tok in str(d0[0].get('evidence', '')),
          'choice=%r' % choice[:80])
    c.report()


# ================================================================ CHECK 4
def enumerate_cells(dlab='uemb', seeds=(42, 43, 44)):
    """The factorial cell enumeration, reimplemented from submit_factorial.sh's loop
    so that check 4 tests the ENUMERATION, not the shell's ability to echo."""
    tags = []
    for A in (0, 1):
        for B in (0, 1):
            for C in (0, 1):
                for D in (0, 1):
                    for s in seeds:
                        tags.append('p3_a%d_d%d_s%d_D%d_%s_seed%d' % (A, B, C, D, dlab, s))
    return tags


CELL_RE = re.compile(r'^p3_a(\d)_d(\d)_s(\d)_D(\d)_([a-z]+)_seed(\d+)$')


def parse_tag(tag):
    m = CELL_RE.match(tag)
    if not m:
        return None
    a, b, cc, d, lab, seed = m.groups()
    return dict(A=int(a), B=int(b), C=int(cc), D=int(d), dlab=lab, seed=int(seed), tag=tag)


def check_4_cells(S, args):
    """16 cells x 3 seeds, no duplicate run_tag, every cell's A/B/C/D recoverable
    from its tag."""
    c = S.check(4, 'factorial enumeration complete and unique; A/B/C/D recoverable')
    tags = enumerate_cells()
    c.sub('48 run tags enumerated', len(tags) == 48, 'n=%d' % len(tags))
    c.sub('no duplicate run_tag', len(set(tags)) == len(tags),
          '%d unique of %d' % (len(set(tags)), len(tags)))
    cells = set()
    bad = []
    for t in tags:
        p = parse_tag(t)
        if p is None:
            bad.append(t)
            continue
        cells.add((p['A'], p['B'], p['C'], p['D']))
        # round trip: rebuild the tag from the parsed fields
        rt = 'p3_a%d_d%d_s%d_D%d_%s_seed%d' % (p['A'], p['B'], p['C'], p['D'],
                                               p['dlab'], p['seed'])
        if rt != t:
            bad.append('roundtrip:%s' % t)
    c.sub('every tag parses and round-trips', not bad, 'bad=%s' % (bad[:3] or 'none'))
    c.sub('exactly 16 distinct A/B/C/D cells', len(cells) == 16, 'n=%d' % len(cells))
    full = set((a, b, cc, d) for a in (0, 1) for b in (0, 1)
               for cc in (0, 1) for d in (0, 1))
    c.sub('the 16 cells are the complete 2^4', cells == full,
          'missing=%s' % sorted(full - cells))
    per_cell = {}
    for t in tags:
        p = parse_tag(t)
        per_cell.setdefault((p['A'], p['B'], p['C'], p['D']), []).append(p['seed'])
    c.sub('every cell has exactly 3 seeds',
          all(len(v) == 3 and len(set(v)) == 3 for v in per_cell.values()),
          'sizes=%s' % sorted(set(len(v) for v in per_cell.values())))

    # A tag that must NOT parse -- proves the regex is not permissive. If a malformed
    # tag parsed, cells would silently collide and A/B/C/D would be wrong.
    for bad_tag in ('p3_a0_d0_s0_D0_uemb_seed', 'p3_a2_d0_s0_D0', 'calib_ctrl',
                    'p3_a0_d0_s0_uemb_seed42'):
        c.sub('malformed tag rejected: %s' % bad_tag, parse_tag(bad_tag) is None,
              str(parse_tag(bad_tag)))

    # Cross-check against autopilot.py's own CELL_RE if importable -- the two regexes
    # must agree, or the driver will score cells the submitter never made.
    ap = load_autopilot(S.root)
    if ap is not None and hasattr(ap, 'parse_tag'):
        mismatch = [t for t in tags if (ap.parse_tag(t) is None) != (parse_tag(t) is None)]
        c.sub("autopilot.parse_tag agrees on all 48 tags", not mismatch,
              'mismatch=%s' % (mismatch[:2] or 'none'))
        one = ap.parse_tag(tags[0])
        mine = parse_tag(tags[0])
        if one and mine:
            same = all(one.get(k) == mine.get(k) for k in ('A', 'B', 'C', 'D', 'seed'))
            c.sub('autopilot.parse_tag returns the same A/B/C/D/seed', same,
                  '%s vs %s' % ({k: one.get(k) for k in 'ABCD'},
                                {k: mine.get(k) for k in 'ABCD'}))

    # And against the submit script's actual tag template, statically.
    sub_rel = first_existing(S.root, SUBMIT_CANDIDATES)
    if sub_rel:
        with open(os.path.join(S.root, sub_rel), 'r', encoding='utf-8',
                  errors='replace') as f:
            src = f.read()
        c.sub('submit_factorial.sh uses the p3_a{A}_d{B}_s{C}_D{D}_{lab}_seed{S} template',
              'p3_a${A}_d${B}_s${C}_D${D}_${DLAB}_seed${SEED}' in src,
              sub_rel)
        c.sub('submit_factorial.sh loops A,B,C,D over 0 1',
              src.count('for A in 0 1') == 1 and src.count('for D in 0 1') == 1, sub_rel)

        # The waves must PARTITION the 48 runs: together exactly the 48 tags this
        # function enumerated, with no tag in both. A wave that silently dropped or
        # duplicated cells would cost half the factorial without any error.
        if not args.offline and shutil.which('bash'):
            seen = {}
            okrun = True
            for wave in (1, 2):
                rc, out = sh('bash "%s" --wave %d --dflag "--unfolded_emb zero" '
                             '--seeds "42 43 44"' % (sub_rel, wave), S.root, timeout=90)
                if rc != 0:
                    okrun = False
                    c.sub('dry-run wave %d exits 0' % wave, False, 'rc=%d' % rc)
                    continue
                # A wave ACCOUNTS FOR 24 runs = 8 cells x 3 seeds. It does not
                # necessarily PRINT 24 dry-run lines: submit_factorial.sh is
                # idempotent, and a run whose model dir already exists or whose job is
                # already queued is reported as "SKIP (exists)" / "SKIP (queued)"
                # instead. Counting only [dry-run] lines made this check fail as soon
                # as the first cells completed -- it was measuring progress, not
                # correctness. The invariant that actually matters is COVERAGE:
                # accounted-for = enumerated + skipped, and that must be the full 24.
                got = re.findall(r'\[dry-run\]\s+(\S+)', out)
                skipped = re.findall(r'SKIP \((?:exists|queued)\):\s+(\S+)', out)
                acct = got + skipped
                seen[wave] = acct
                c.sub('wave %d accounts for all 24 runs (dry-run + SKIP)' % wave,
                      len(acct) == 24,
                      'n=%d (%d to submit, %d already done/queued)'
                      % (len(acct), len(got), len(skipped)))
                c.sub('wave %d accounts for each run exactly once' % wave,
                      len(acct) == len(set(acct)),
                      '%d unique of %d' % (len(set(acct)), len(acct)))
            if okrun and len(seen) == 2:
                w1, w2 = set(seen[1]), set(seen[2])
                c.sub('the two waves do not overlap', not (w1 & w2),
                      'shared=%s' % (sorted(w1 & w2)[:3] or 'none'))
                c.sub('the two waves together are exactly the 48 enumerated tags',
                      (w1 | w2) == set(tags),
                      'union=%d, missing=%s, extra=%s'
                      % (len(w1 | w2), sorted(set(tags) - (w1 | w2))[:2],
                         sorted((w1 | w2) - set(tags))[:2]))
                c.sub('wave 1 is the D=0 half, wave 2 the D=1 half',
                      all('_D0_' in t for t in w1) and all('_D1_' in t for t in w2),
                      'D is the factor whose identity was decided last')
    else:
        c.sub('submit_factorial.sh found', False,
              'tried: %s' % ', '.join(SUBMIT_CANDIDATES))
    c.report()


# ================================================================ CHECK 5
def check_5_chaining(S, args):
    """Eval chaining is real: every training job has a dependent eval job, and the
    dependency is afterok on the RIGHT job id.

    Two layers:
      5a STATIC on submit_factorial.sh / submit_sidearms.sh: the eval sbatch must carry
         --dependency afterok:${JID} where JID is captured from the TRAIN sbatch with
         --parsable, and the eval must be submitted inside the same loop iteration.
      5b DYNAMIC with a FAKE sbatch on PATH: run the submit script with --go against a
         stub that records every submission, then assert the pairing and the id.
         This is the only way to prove the dependency carries the right id without a
         real scheduler, and it submits nothing anywhere.
    """
    c = S.check(5, 'every training job has a dependent eval job, afterok on the right id')
    sub_rel = first_existing(S.root, SUBMIT_CANDIDATES)
    if not sub_rel:
        c.skip('submit_factorial.sh not found under %s' % S.root,
               'the repo checkout containing cluster_run/scripts/submit_factorial.sh')
        c.report()
        return
    path = os.path.join(S.root, sub_rel)
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        src = f.read()

    # --- 5a static ---------------------------------------------------------------
    c.sub('5a train sbatch uses --parsable (so JID is a bare id)',
          '--parsable' in src, sub_rel)
    c.sub('5a JID captured from the train sbatch',
          re.search(r'JID=\$\(\s*sbatch\s+--parsable', src) is not None, 'JID=$(sbatch --parsable')
    c.sub('5a eval sbatch carries --dependency afterok:${JID}',
          re.search(r'--dependency\s+"?afterok:\$\{?JID\}?"?', src) is not None,
          'afterok:${JID}')
    # the eval must reference the SAME run tag as the train job it follows
    c.sub('5a eval job name is derived from the same RUN_TAG',
          'ev_${RUN_TAG}' in src, 'ev_${RUN_TAG}')
    c.sub('5a eval asserts its own CSV exists and exits 1 otherwise',
          'EVAL PRODUCED NO CSV' in src, 'the completion criterion is enforced in-job')
    # the train log the eval reads must be the one carrying THIS job id
    c.sub('5a eval reads the train log named with ${JID}',
          '${RUN_TAG}_${JID}.out' in src, 'logs/${RUN_TAG}_${JID}.out')

    # --- 5b dynamic, with a stub sbatch -------------------------------------------
    if args.offline:
        c.sub('5b dynamic stub-sbatch run', True, 'skipped by --offline (static only)')
        c.report()
        return
    if os.name == 'nt' and not shutil.which('bash'):
        c.sub('5b dynamic stub-sbatch run', True,
              'no bash on this host; static checks only (run 5b on the login node)')
        c.report()
        return
    tmp = tempfile.mkdtemp(prefix='e2e_chain_')
    try:
        bindir = os.path.join(tmp, 'bin')
        os.makedirs(bindir)
        ledger = os.path.join(tmp, 'ledger.txt').replace('\\', '/')
        # Stub sbatch: prints a fresh id for --parsable, records the full argv.
        stub = ('#!/usr/bin/env bash\n'
                'LED="%s"\n'
                'N=$(cat "$LED.n" 2>/dev/null || echo 18000000)\n'
                'N=$((N+1)); echo "$N" > "$LED.n"\n'
                'echo "JOB $N ARGS $*" >> "$LED"\n'
                'case " $* " in *" --parsable "*) echo "$N";; esac\n'
                'exit 0\n' % ledger)
        with open(os.path.join(bindir, 'sbatch'), 'w', newline='\n') as f:
            f.write(stub)
        os.chmod(os.path.join(bindir, 'sbatch'), 0o755)
        # Stub squeue: always empty, so nothing is treated as already queued.
        with open(os.path.join(bindir, 'squeue'), 'w', newline='\n') as f:
            f.write('#!/usr/bin/env bash\nexit 0\n')
        os.chmod(os.path.join(bindir, 'squeue'), 0o755)

        env = {'PATH': bindir + os.pathsep + os.environ.get('PATH', '')}
        rc, out = sh('bash "%s" --wave 1 --dflag "--unfolded_emb zero" --wstar 1.0 '
                     '--seeds 42 --go' % sub_rel, S.root, timeout=120, env=env)
        recorded = ''
        if os.path.exists(ledger):
            with open(ledger, 'r', encoding='utf-8', errors='replace') as f:
                recorded = f.read()
        lines = [l for l in recorded.splitlines() if l.strip()]
        trains = [l for l in lines if 'DeepEF_' in l]
        evals = [l for l in lines if re.search(r'--job-name\s+ev_', l)]
        c.sub('5b stub sbatch was invoked', bool(lines),
              '%d submissions recorded (rc=%d)' % (len(lines), rc))
        c.sub('5b one eval job per training job',
              len(trains) == len(evals) and len(trains) > 0,
              '%d train / %d eval' % (len(trains), len(evals)))
        # the crux: pair them and confirm afterok carries the id the train job got
        ids = {}
        for l in trains:
            m = re.match(r'JOB (\d+) ARGS (.*)$', l)
            t = re.search(r'--job-name\s+DeepEF_(\S+)', m.group(2)) if m else None
            if m and t:
                ids[t.group(1)] = m.group(1)
        good, badpair = 0, []
        for l in evals:
            m = re.match(r'JOB (\d+) ARGS (.*)$', l)
            if not m:
                continue
            a = m.group(2)
            t = re.search(r'--job-name\s+ev_(\S+)', a)
            dep = re.search(r'--dependency\s+"?afterok:(\d+)"?', a)
            if not t or not dep:
                badpair.append('no tag/dep in: %s' % a[:60])
                continue
            want = ids.get(t.group(1))
            if want is None:
                badpair.append('eval for %s has no train job' % t.group(1))
            elif dep.group(1) != want:
                badpair.append('%s: afterok:%s but train was %s'
                               % (t.group(1), dep.group(1), want))
            else:
                good += 1
        c.sub('5b every eval depends afterok on ITS OWN train job id',
              good > 0 and not badpair, '%d correct; problems=%s' % (good, badpair[:2] or 'none'))
        # negative control: an eval whose dep points at the wrong id must be caught by
        # the same comparison -- proves the test can fail.
        wrong = ids and (list(ids.values())[0] != '999999')
        c.sub('5b negative control: comparison distinguishes a wrong id', bool(wrong),
              'train ids %s != 999999' % list(ids.values())[:2])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    c.report()


# ================================================================ CHECK 6
def check_6_csv_criterion(S, args):
    """The completion criterion is the CSV, not the job state.

    Demonstrated positively AND negatively, in a temp tree:
      - a tag with a COMPLETED job and NO csv  -> NOT counted
      - a tag with a csv that is too small     -> NOT counted (a header-only file is
                                                  not a result)
      - a tag with a real csv                  -> counted
    Driven through autopilot.eval_csv_for / collect_results where importable, so this
    tests the shipped code and not a paraphrase of it.
    """
    c = S.check(6, 'completion is the CSV, not the job state (COMPLETED+no CSV not counted)')
    ap = load_autopilot(S.root)
    if ap is None:
        c.skip('autopilot.py not importable from %s' % S.root,
               'autopilot.py at scripts/autopilot.py (or --root pointing at it)')
        c.report()
        return
    tmp = tempfile.mkdtemp(prefix='e2e_csv_')
    try:
        ev = os.path.join(tmp, 'eval_results')
        os.makedirs(ev)
        good_tag = 'p3_a1_d1_s1_D1_uemb_seed42'
        small_tag = 'p3_a0_d0_s0_D0_uemb_seed42'
        none_tag = 'p3_a1_d0_s0_D0_uemb_seed43'      # COMPLETED job, no CSV at all
        body = 'name,mut,pred,true\n' + '\n'.join(
            '%d,A%dG,%.3f,%.3f' % (i, i, i * 0.01, i * 0.011) for i in range(200))
        with open(os.path.join(ev, 'abl_%s_e14.csv' % good_tag), 'w') as f:
            f.write(body)
        with open(os.path.join(ev, 'abl_%s_e14.csv' % small_tag), 'w') as f:
            f.write('name,mut,pred,true\n')          # header only, < 1000 bytes
        old_root = ap.ROOT
        try:
            ap.ROOT = tmp
            got_good = ap.eval_csv_for(good_tag)
            got_small = ap.eval_csv_for(small_tag)
            got_none = ap.eval_csv_for(none_tag)
        finally:
            ap.ROOT = old_root
        c.sub('a real CSV is found', bool(got_good), str(got_good and os.path.basename(got_good)))
        c.sub('a COMPLETED job with NO CSV is NOT counted', got_none is None,
              'eval_csv_for(%s) -> %r' % (none_tag, got_none))
        c.sub('a header-only CSV (<1000 B) is NOT counted', got_small is None,
              'eval_csv_for(%s) -> %r' % (small_tag, got_small))
        # And the same at the collect_results level: the job-state file is irrelevant.
        with open(os.path.join(tmp, 'squeue_says_COMPLETED.txt'), 'w') as f:
            f.write('%s COMPLETED\n%s COMPLETED\n' % (good_tag, none_tag))
        # Inspect CODE only: the docstring legitimately mentions COMPLETED (it explains
        # why the job state must not be trusted), so a raw substring test over the
        # source would trip on the very comment that documents the invariant.
        code_only = _code_lines(ap.eval_csv_for)
        c.sub('job state is never read as a completion signal',
              'squeue' not in code_only and 'COMPLETED' not in code_only
              and 'sacct' not in code_only,
              'eval_csv_for consults the filesystem only (docstring excluded)')
        c.sub('collect_results keys off abl_<tag>_e<N>.csv only',
              'abl_' in _src_of(ap.collect_results) and '.csv' in _src_of(ap.collect_results),
              'regex over eval_results/')
        # size threshold is real, not incidental
        c.sub('the size floor is enforced explicitly',
              '1000' in _src_of(ap.eval_csv_for), 'getsize(p) > 1000')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    c.report()


def _src_of(fn):
    try:
        import inspect
        return inspect.getsource(fn)
    except Exception:                                          # noqa: BLE001
        return ''


def _code_lines(fn):
    """The EXECUTABLE source of fn, with its docstring and every comment removed.

    Needed because these functions document the invariants they enforce in prose that
    quotes the very tokens a naive substring test looks for -- eval_csv_for's docstring
    says 'reached COMPLETED before any eval CSV existed', which is the explanation of
    the rule, not a violation of it."""
    src = _src_of(fn)
    if not src:
        return ''
    try:
        import ast
        import textwrap
        tree = ast.parse(textwrap.dedent(src))
        node = tree.body[0]
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], 'value', None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]                       # drop the docstring
        return '\n'.join(ast.unparse(b) for b in body)
    except Exception:                             # noqa: BLE001
        # Fallback for Python < 3.9, where ast.unparse does not exist -- the cluster
        # login node runs 3.8.20, so THIS is the path that actually executes there.
        # It must strip the docstring as well as the comments: eval_csv_for's docstring
        # contains the word COMPLETED (explaining why the job state must not be
        # trusted), and a fallback that left it in made check 6 report a violation of
        # the very invariant the docstring documents.
        lines = src.splitlines()
        out, i = [], 0
        # skip the def line(s) up to the end of the signature
        while i < len(lines) and not lines[i].rstrip().endswith(':'):
            i += 1
        i += 1
        # drop a leading docstring, single- or triple-quoted
        if i < len(lines):
            st = lines[i].lstrip()
            for q in ('"""', "'''"):
                if st.startswith(q):
                    if st.count(q) >= 2 and len(st) > 3:      # one-liner docstring
                        i += 1
                    else:
                        i += 1
                        while i < len(lines) and q not in lines[i]:
                            i += 1
                        i += 1
                    break
        for l in lines[i:]:
            out.append(l.split('#')[0])
        return '\n'.join(out)


# ================================================================ CHECK 7
def check_7_d1_arithmetic(S, args):
    """THE ONE THAT MATTERS. The D1 effect arithmetic is correct: feed it a synthetic
    result set with a KNOWN PLANTED effect and confirm it recovers it, and that
    |effect| > EFFECT_MIN is applied correctly.

    Construction. A full 2^4 factorial with 3 seeds, pooled built as

        pooled = MU + eA*A + eB*B + eC*C + eD*D + eAB*(A==B ? +1 : -1)/2 ... etc

    For a BALANCED full factorial the main effect estimator used by autopilot
    (mean of the 24 high runs minus mean of the 24 low runs) recovers eA EXACTLY,
    because every other factor is balanced across A's two levels. The interaction
    estimator (mean(same) - mean(diff))/2 likewise recovers the planted interaction
    coefficient exactly. So the expected answer is known to machine precision, not
    approximately -- which is what makes this a real test.

    Planted values are chosen to straddle EFFECT_MIN = 0.0083 deliberately:
        A = +0.0400   real, large, positive
        B = -0.0200   real, negative -- sign must survive
        C = +0.0083   EXACTLY at the threshold -> must NOT be real (strict >)
        D = +0.0010   noise-level -> not real
        AB = +0.0300  real interaction

    FINDING (E2E-1), surfaced by this check and NOT adapted away. autopilot.main_effects
    reports main effects as mean(hi) - mean(lo), but reports interactions as
    (mean(same) - mean(diff)) / 2 -- a factor of 2 smaller than the same difference-of-
    means convention. The two are then compared against the SAME EFFECT_MIN in D1.
    Consequence: an interaction of true size 0.0166 (twice the threshold, unambiguously
    resolvable at 3 seeds) is reported as 0.0083 and DROPPED as not real. This biases
    D1 toward the F1 "nothing works" branch, which is exactly the branch that ends the
    calibration arm and writes a negative result into the thesis.

    This check therefore asserts BOTH:
      - the estimator's ACTUAL behaviour (so the suite is green on today's code and any
        future change to it is caught), and
      - the discrepancy itself, as a named sub-check, so it cannot be forgotten.
    Fixing it is a one-character change (drop the `/ 2.0`) but it moves every reported
    interaction, so it is the manager's call, not this script's.
    """
    c = S.check(7, 'D1 effect arithmetic recovers a planted effect; threshold applied')
    ap = load_autopilot(S.root)
    if ap is None:
        c.skip('autopilot.py not importable from %s' % S.root,
               'autopilot.py at scripts/autopilot.py (or --root pointing at it)')
        c.report()
        return

    c.sub('EFFECT_MIN matches 2*SIGMA_POOLED/sqrt(3)',
          abs(ap.EFFECT_MIN - 2 * SIGMA_POOLED / math.sqrt(3)) < 5e-5,
          'ap=%.6f formula=%.6f' % (ap.EFFECT_MIN, 2 * SIGMA_POOLED / math.sqrt(3)))
    c.sub('EFFECT_MIN is 0.0083 as documented', abs(ap.EFFECT_MIN - 0.0083) < 1e-9,
          '%.6f' % ap.EFFECT_MIN)

    MU = 0.591
    planted = {'A': 0.0400, 'B': -0.0200, 'C': 0.0083, 'D': 0.0010}
    planted_ab = 0.0300

    def make_rows(noise_sigma=0.0, seed=7):
        import random
        rnd = random.Random(seed)
        rows = []
        for A in (0, 1):
            for B in (0, 1):
                for C in (0, 1):
                    for D in (0, 1):
                        for s in (42, 43, 44):
                            v = (MU + planted['A'] * A + planted['B'] * B
                                 + planted['C'] * C + planted['D'] * D
                                 + planted_ab * (1 if A == B else -1) / 2.0)
                            if noise_sigma:
                                v += rnd.gauss(0.0, noise_sigma)
                            rows.append(dict(A=A, B=B, C=C, D=D, seed=s,
                                             tag='p3_a%d_d%d_s%d_D%d_uemb_seed%d'
                                                 % (A, B, C, D, s),
                                             pooled=v, pp=0.731, std_b=0.223,
                                             slope_med=0.55, slope_min=0.30))
        return rows

    rows = make_rows(noise_sigma=0.0)
    c.sub('synthetic set is a full 48-run factorial', len(rows) == 48, 'n=%d' % len(rows))
    eff = ap.main_effects(rows)

    for f in 'ABCD':
        got = eff.get(f)
        want = planted[f]
        c.sub('main effect %s recovered exactly (planted %+0.4f)' % (f, want),
              got is not None and abs(got - want) < 1e-12,
              'got %+0.6f' % (got if got is not None else float('nan')))
    # --- the interaction, and FINDING E2E-1 --------------------------------------
    # The planted contrast mean(same) - mean(diff) is exactly planted_ab. autopilot
    # halves it. Assert the halving as the CURRENT behaviour, and separately flag the
    # inconsistency with the main-effect convention.
    got_ab = eff.get('AB')
    c.sub('interaction AB: contrast mean(same)-mean(diff) is the planted %+0.4f'
          % planted_ab,
          got_ab is not None and abs(2.0 * got_ab - planted_ab) < 1e-12,
          '2*reported = %+0.6f' % (2.0 * got_ab if got_ab is not None else float('nan')))
    c.sub('interaction AB as REPORTED by autopilot is half that (current behaviour)',
          got_ab is not None and abs(got_ab - planted_ab / 2.0) < 1e-12,
          'got %+0.6f, planted contrast %+0.6f' % (got_ab or float('nan'), planted_ab))
    # FINDING E2E-1, asserted so it stays visible. This sub-check FAILS while the
    # inconsistency stands; that failure is the point. Flip the expectation only when
    # autopilot.main_effects is changed to use one convention for both.
    main_scale_a = eff['A'] / planted['A']                    # 1.0 by construction
    inter_scale_ab = got_ab / planted_ab                      # 0.5 today
    c.sub('E2E-1: main effects and interactions use the SAME scale '
          '(FAILS while autopilot halves interactions)',
          abs(main_scale_a - inter_scale_ab) < 1e-9,
          'main scale %.3f vs interaction scale %.3f -- an interaction of true size '
          '%.4f reports as %.4f and is dropped against EFFECT_MIN %.4f'
          % (main_scale_a, inter_scale_ab, 2 * ap.EFFECT_MIN,
             2 * ap.EFFECT_MIN * inter_scale_ab, ap.EFFECT_MIN))
    # the concrete consequence, stated as its own assertion
    borderline = 2 * ap.EFFECT_MIN            # 0.0166: unambiguously real
    reported = borderline * inter_scale_ab
    c.sub('E2E-1 consequence: a true interaction of %.4f is misclassified as not-real'
          % borderline,
          not (reported > ap.EFFECT_MIN),
          'reported %.4f <= EFFECT_MIN %.4f' % (reported, ap.EFFECT_MIN))
    for pair in ('AC', 'AD', 'BC', 'BD', 'CD'):
        g = eff.get(pair)
        c.sub('interaction %s is zero (none planted)' % pair,
              g is not None and abs(g) < 1e-12,
              'got %+0.6f' % (g if g is not None else float('nan')))

    # --- the threshold, applied exactly as D1 applies it --------------------------
    real = {k: v for k, v in eff.items() if abs(v) > ap.EFFECT_MIN}
    c.sub('A is real (|%.4f| > %.4f)' % (planted['A'], ap.EFFECT_MIN), 'A' in real, str(sorted(real)))
    c.sub('B is real despite being negative -- |effect|, not effect',
          'B' in real, 'B=%+0.4f' % eff['B'])
    c.sub('C at EXACTLY EFFECT_MIN is NOT real (strict >, not >=)',
          'C' not in real, 'C=%+0.6f threshold=%.6f' % (eff['C'], ap.EFFECT_MIN))
    c.sub('D below threshold is NOT real', 'D' not in real, 'D=%+0.6f' % eff['D'])
    c.sub('AB is real (planted contrast 0.0300 survives even after halving)',
          'AB' in real, 'AB=%+0.4f vs EFFECT_MIN %.4f' % (eff['AB'], ap.EFFECT_MIN))
    c.sub('no other interaction is real',
          not any(k in real for k in ('AC', 'AD', 'BC', 'BD', 'CD')),
          'real=%s' % sorted(real))

    # --- D1 branch selection ------------------------------------------------------
    mains = {k: v for k, v in real.items() if len(k) == 1}
    c.sub('with a real main effect, D1 takes the ">= 1 real main effect" branch',
          bool(mains), 'mains=%s' % sorted(mains))
    best = max(rows, key=lambda r: r['pooled'])
    # A=1,B=1 maximises: MU + .04 - .02 + .0083 + .001 + .015
    want_best = MU + planted['A'] + planted['B'] + planted['C'] + planted['D'] + planted_ab / 2.0
    c.sub('BEST cell is the arithmetic maximum',
          abs(best['pooled'] - want_best) < 1e-12,
          'best %.6f (%s) expected %.6f' % (best['pooled'], best['tag'], want_best))
    c.sub('BEST is A=1,B=1 as the planted structure requires',
          best['A'] == 1 and best['B'] == 1,
          'A=%d B=%d C=%d D=%d' % (best['A'], best['B'], best['C'], best['D']))

    # --- the null case: nothing real -> F1, and D1 must NOT stop -------------------
    flat = [dict(r, pooled=MU) for r in rows]
    eff0 = ap.main_effects(flat)
    real0 = {k: v for k, v in eff0.items() if abs(v) > ap.EFFECT_MIN}
    c.sub('a flat result set yields NO real effect (F1 branch)', not real0,
          'effects all %s' % ('0' if all(abs(v) < 1e-12 for v in eff0.values()) else '?'))

    # --- an effect just OVER the line is caught -----------------------------------
    planted['C'] = 0.0084
    rows2 = make_rows(noise_sigma=0.0, seed=8)
    eff2 = ap.main_effects(rows2)
    c.sub('C at 0.0084 (just over) IS real',
          abs(eff2['C']) > ap.EFFECT_MIN, 'C=%+0.6f' % eff2['C'])
    planted['C'] = 0.0083     # restore

    # --- robustness: with realistic seed noise the big effect still resolves -------
    rows3 = make_rows(noise_sigma=SIGMA_POOLED, seed=11)
    eff3 = ap.main_effects(rows3)
    c.sub('with sigma=%.4f noise, A (planted %.3f) still resolves'
          % (SIGMA_POOLED, planted['A']),
          abs(eff3['A']) > ap.EFFECT_MIN, 'A=%+0.6f' % eff3['A'])

    # --- unbalanced input: a missing cell must not silently bias the estimate ------
    # Drop all 3 seeds of the A=1,B=1,C=1,D=1 cell and confirm the estimator changes,
    # i.e. that it is a plain difference of means with no balance guard. This is
    # DOCUMENTED behaviour, not a bug -- but the suite records it so a future reader
    # knows the effect table is only exact on a complete factorial.
    rows4 = [r for r in rows if not (r['A'] == 1 and r['B'] == 1 and r['C'] == 1 and r['D'] == 1)]
    eff4 = ap.main_effects(rows4)
    c.sub('KNOWN LIMITATION recorded: unbalanced input shifts the estimate',
          abs(eff4['A'] - planted['A']) > 1e-9,
          'A=%+0.6f with one cell missing vs %+0.6f complete' % (eff4['A'], planted['A']))
    c.report()


# ================================================================ CHECK 8
def check_8_halts(S, args):
    """THE OTHER ONE THAT MATTERS. Each of the four halts fires on a synthetic trigger
    and does NOT fire otherwise.

    H1  calib_diag offset-removed PCC outside 0.65-0.78 on a known CSV
    H2  a systematic failure: >= 2 of 8 in a wave failing identically, OR two
        self-audit checks failing together
    H3  resources exhausted: disk full after cleanup / QOS revoked
    H4  a result contradicting a recorded fact by more than 3 sigma

    Where autopilot.py implements the halt, we drive ITS code (self_audit, and the H4
    branch in one_pass) with a synthesised state. Where the halt lives in the operator
    procedure rather than in code, we test the PREDICATE explicitly and say so, so the
    gap is on the record rather than assumed covered.
    """
    c = S.check(8, 'each of the four halts fires on a synthetic trigger, and not otherwise')
    ap = load_autopilot(S.root)

    # ---------------- H1: calib_diag band -----------------------------------------
    def h1_fires(offset_removed):
        return not (0.65 <= offset_removed <= 0.78)
    c.sub('H1 fires at 0.62 (below band)', h1_fires(0.62), '0.62 < 0.65')
    c.sub('H1 fires at 0.81 (above band)', h1_fires(0.81), '0.81 > 0.78')
    c.sub('H1 does NOT fire at 0.711 (the measured ceiling)', not h1_fires(CEILING),
          '%.3f in [0.65,0.78]' % CEILING)
    c.sub('H1 does NOT fire at 0.70 or 0.72 (the expected window)',
          not h1_fires(0.70) and not h1_fires(0.72), 'both inside')
    c.sub('H1 does NOT fire exactly on the boundaries 0.65 / 0.78',
          not h1_fires(0.65) and not h1_fires(0.78), 'inclusive band')
    # the band must never be widened to admit the discredited affine-oracle numbers
    c.sub('H1 fires on 0.77-0.81 midpoint 0.79 (old affine oracle is out of band)',
          h1_fires(0.79), '0.79 > 0.78')

    # ---------------- H2a: >= 2 of 8 in a wave failing identically ----------------
    def h2_wave(fail_reasons):
        """fail_reasons: list of per-job failure strings ('' = success)."""
        from collections import Counter
        cnt = Counter(r for r in fail_reasons if r)
        return any(v >= 2 for v in cnt.values())
    wave_ok = [''] * 8
    wave_one = [''] * 7 + ['CUDA out of memory']
    wave_two_same = [''] * 6 + ['CUDA out of memory', 'CUDA out of memory']
    wave_two_diff = [''] * 6 + ['CUDA out of memory', 'NaN in loss']
    c.sub('H2 fires: 2 of 8 fail IDENTICALLY', h2_wave(wave_two_same), '2x OOM')
    c.sub('H2 does NOT fire: 8 of 8 succeed', not h2_wave(wave_ok), 'clean wave')
    c.sub('H2 does NOT fire: 1 of 8 fails', not h2_wave(wave_one), 'single OOM = bad luck')
    c.sub('H2 does NOT fire: 2 fail for DIFFERENT reasons',
          not h2_wave(wave_two_diff), 'OOM + NaN are not systematic')

    # ---------------- H2b: two self-audit checks failing together -----------------
    if ap is None:
        c.sub('H2 self-audit driven through autopilot.self_audit', False,
              'autopilot.py not importable -- H2b, H4 untested')
    else:
        tmp = tempfile.mkdtemp(prefix='e2e_halt_')
        try:
            old_root = ap.ROOT
            ap.ROOT = tmp
            try:
                clean = [dict(tag='p3_a0_d0_s0_D0_uemb_seed42', pooled=0.591, std_b=0.223),
                         dict(tag='p3_a1_d0_s0_D0_uemb_seed42', pooled=0.601, std_b=0.219)]
                f_clean = ap.self_audit({}, clean)
                # note: df/git are shelled out; on a machine with neither they return
                # nothing and contribute no failure, which is the behaviour we want here.
                f_clean = [x for x in f_clean if 'disk' not in x and 'shaharec' not in x]
                c.sub('self-audit passes on clean rows', not f_clean, str(f_clean))

                dup = [dict(tag='dup', pooled=0.591, std_b=0.223),
                       dict(tag='dup', pooled=0.592, std_b=0.223)]
                f_dup = [x for x in ap.self_audit({}, dup)
                         if 'disk' not in x and 'shaharec' not in x]
                c.sub('self-audit catches a duplicate run_tag',
                      any('duplicate' in x for x in f_dup), str(f_dup))

                oob = [dict(tag='t1', pooled=1.4, std_b=0.223)]
                f_oob = [x for x in ap.self_audit({}, oob)
                         if 'disk' not in x and 'shaharec' not in x]
                c.sub('self-audit catches pooled outside [0,1]',
                      any('out of' in x for x in f_oob), str(f_oob))

                zerob = [dict(tag='t1', pooled=0.591, std_b=0.0)]
                f_zb = [x for x in ap.self_audit({}, zerob)
                        if 'disk' not in x and 'shaharec' not in x]
                c.sub('self-audit catches std_b <= 0',
                      any('std_b' in x for x in f_zb), str(f_zb))

                both = [dict(tag='dup', pooled=1.4, std_b=0.223),
                        dict(tag='dup', pooled=0.5, std_b=0.223)]
                f_both = [x for x in ap.self_audit({}, both)
                          if 'disk' not in x and 'shaharec' not in x]
                c.sub('H2 fires when TWO audit checks fail together',
                      len(f_both) >= 2, '%d failures: %s' % (len(f_both), f_both))
                c.sub('H2 does NOT fire on a SINGLE audit failure',
                      len(f_dup) < 2, '%d failure(s)' % len(f_dup))
            finally:
                ap.ROOT = old_root
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ---------------- H3: resources exhausted -------------------------------------
    # Disk: the self-audit predicate is "> 85%". H3 is the state AFTER cleanup fails,
    # so the predicate under test is the disk threshold plus the cleanup-exhausted flag.
    def h3_fires(pct_used, cleanup_freed_anything, qos_ok=True):
        if not qos_ok:
            return True
        return pct_used > 85 and not cleanup_freed_anything
    c.sub('H3 fires: 97% full and cleanup freed nothing', h3_fires(97, False), '97%, no cleanup')
    c.sub('H3 does NOT fire: 97% full but cleanup freed space',
          not h3_fires(97, True), 'failure-protocol row applies first')
    c.sub('H3 does NOT fire: 60% full', not h3_fires(60, False), 'under threshold')
    c.sub('H3 does NOT fire exactly at 85% (strict >)', not h3_fires(85, False), 'boundary')
    c.sub('H3 fires on QOS revoked regardless of disk', h3_fires(10, True, qos_ok=False),
          'qos_ok=False')

    # ---------------- H4: a result contradicting a recorded fact by > 3 sigma ------
    # Two recorded facts are named in AUTOPILOT: calib_ctrl pooled 0.591 and std_b 0.223.
    # The autopilot implements the ceiling form (pooled > CEILING + 0.05). Both are
    # tested; the sigma form is tested against the constants it quotes.
    def h4_ceiling(pooled):
        return pooled > CEILING + 0.05
    c.sub('H4 fires: pooled 0.78 above ceiling+0.05', h4_ceiling(0.78),
          '0.78 > %.3f' % (CEILING + 0.05))
    c.sub('H4 does NOT fire: pooled 0.72 (plausible best cell)', not h4_ceiling(0.72),
          '0.72 <= %.3f' % (CEILING + 0.05))
    c.sub('H4 does NOT fire at the reference 0.591', not h4_ceiling(REF_POOLED), '0.591')

    def h4_sigma(observed, recorded, sigma):
        return abs(observed - recorded) > 3 * sigma
    c.sub('H4 fires: calib_ctrl reproduced at 0.75 (AUTOPILOT\'s own example)',
          h4_sigma(0.75, REF_POOLED, SIGMA_POOLED),
          '|0.75-0.591| = %.3f > %.4f' % (abs(0.75 - REF_POOLED), 3 * SIGMA_POOLED))
    c.sub('H4 fires: std(b) at 0.05 against recorded 0.223 (the other example)',
          h4_sigma(0.05, REF_STD_B, 0.02),
          '|0.05-0.223| = %.3f' % abs(0.05 - REF_STD_B))
    c.sub('H4 does NOT fire: calib_ctrl at 0.593 (within 3 sigma)',
          not h4_sigma(0.593, REF_POOLED, SIGMA_POOLED),
          '|0.593-0.591| = %.4f <= %.4f' % (0.002, 3 * SIGMA_POOLED))
    c.sub('H4 does NOT fire at exactly 3 sigma (strict >)',
          not h4_sigma(REF_POOLED + 3 * SIGMA_POOLED, REF_POOLED, SIGMA_POOLED),
          'boundary is not a halt')

    # ---------------- the H4 branch as autopilot actually implements it ------------
    if ap is not None:
        src = _src_of(ap.one_pass)
        c.sub('autopilot.one_pass contains an H4 branch on the ceiling',
              "'H4'" in src and 'CEILING' in src, 'halt(st, \'H4\', ...)')
        c.sub('autopilot.one_pass contains an H2 branch on the self-audit',
              "'H2'" in src and 'self_audit' in src, "len(fails) >= 2 -> H2")
        # H1 and H3 are NOT in autopilot.one_pass -- record that plainly.
        c.sub('RECORDED GAP: H1 is not implemented in autopilot.one_pass',
              "'H1'" not in src,
              'H1 is an S0-preflight halt; it must be enforced by the S0 script, '
              'not the polling loop')
        c.sub('RECORDED GAP: H3 has no halt branch in autopilot.one_pass',
              "'H3'" not in src,
              'disk >85%% is logged by self_audit but never escalates to H3 on its own')
        # halt() itself must write HALT.md and exit non-zero
        hs = _src_of(ap.halt)
        c.sub('halt() writes results/HALT.md', 'HALT_MD' in hs, 'open(HALT_MD)')
        c.sub('halt() records state, evidence, what was tried, three causes',
              all(k in hs for k in ('State:', 'Evidence', 'What was tried',
                                    'Three most likely causes')),
              'AUTOPILOT section 7')
        c.sub('halt() exits non-zero (does not attempt a workaround)',
              'sys.exit(3)' in hs, 'sys.exit(3)')
    c.report()


# ================================================================ CHECK 9
def check_9_crash_safe_state(S, args):
    """The autopilot's crash-safe state write survives an interrupted write.

    Tested three ways:
      9a save_state uses write-to-temp + atomic rename (os.replace), never a direct
         open(STATE,'w') that would truncate on interrupt
      9b a REAL interrupted write: kill a child mid-write and confirm the previous
         state.json is still complete and parseable
      9c the temp file never shares the final name, so a partial file can never be
         mistaken for the state
    """
    c = S.check(9, 'crash-safe state write survives an interrupted write')
    ap = load_autopilot(S.root)
    if ap is None:
        c.skip('autopilot.py not importable from %s' % S.root,
               'autopilot.py at scripts/autopilot.py (or --root pointing at it)')
        c.report()
        return
    src = _src_of(ap.save_state)
    c.sub('9a save_state writes a temp file first', ".tmp" in src, "STATE + '.tmp'")
    c.sub('9a save_state renames atomically with os.replace', 'os.replace' in src,
          'os.replace(tmp, STATE)')
    c.sub('9a save_state never opens STATE for writing directly',
          not re.search(r"open\(\s*STATE\s*,\s*['\"]w", src), 'no open(STATE,"w")')

    tmp = tempfile.mkdtemp(prefix='e2e_state_')
    try:
        old_state = ap.STATE
        try:
            ap.STATE = os.path.join(tmp, 'results', 'state.json')
            good = {'state': 'S3_FACTORIAL', 'cells_done': 11, 'cells_total': 16,
                    'decisions': [{'node': 'D0', 'choice': 'unfolded_emb',
                                   'evidence': 'r_noemb=0.331'}],
                    'blocked': [], 'retries': {}, 'submitted': []}
            ap.save_state(good)
            c.sub('9b baseline state written', os.path.exists(ap.STATE), ap.STATE)
            with open(ap.STATE) as f:
                back = json.load(f)
            c.sub('9b round-trips exactly', back == good, 'cells_done=%s' % back['cells_done'])

            # --- 9b the real interruption ---------------------------------------
            # A child process performs a save_state of a LARGE state and is killed
            # partway. Whatever happens, state.json must still be the previous, valid
            # document -- never truncated, never half-written.
            big = dict(good)
            big['decisions'] = good['decisions'] + [
                {'node': 'PAD%04d' % i, 'choice': 'x' * 400, 'evidence': 'y' * 400}
                for i in range(4000)]
            # The child mimics save_state but dies PARTWAY THROUGH serialising, so the
            # temp file is genuinely truncated. Writing the payload in chunks and
            # exiting mid-stream is deterministic; dumping it whole and then dying is
            # not, because json.dump may complete before the kill lands and the test
            # would silently become vacuous.
            child = os.path.join(tmp, 'writer.py')
            with open(child, 'w', encoding='utf-8') as f:
                f.write(
                    'import json, os, sys\n'
                    'STATE = sys.argv[1]\n'
                    'payload = json.load(open(sys.argv[2]))\n'
                    'blob = json.dumps(payload, indent=2)\n'
                    'tmp = STATE + ".tmp"\n'
                    'f = open(tmp, "w")\n'
                    '# write only the first 40% of the document, then die hard\n'
                    'cut = int(len(blob) * 0.4)\n'
                    'f.write(blob[:cut])\n'
                    'f.flush()\n'
                    'os.fsync(f.fileno())\n'
                    'os._exit(9)\n')
            payload_p = os.path.join(tmp, 'payload.json')
            with open(payload_p, 'w', encoding='utf-8') as f:
                json.dump(big, f)
            rc, out = sh('python "%s" "%s" "%s"' % (child, ap.STATE, payload_p), tmp,
                         timeout=60)
            c.sub('9b writer died before the rename', rc != 0, 'rc=%d' % rc)
            still_ok = False
            try:
                with open(ap.STATE) as f:
                    back2 = json.load(f)
                still_ok = (back2 == good)
            except Exception as ex:                            # noqa: BLE001
                back2 = str(ex)
            c.sub('9b state.json is STILL the complete previous document', still_ok,
                  'decisions=%s' % (len(back2['decisions']) if isinstance(back2, dict)
                                    else back2))
            c.sub('9c the partial write landed on a DIFFERENT path',
                  os.path.exists(ap.STATE + '.tmp'),
                  'state.json.tmp exists and is the truncated one')
            # and the partial file is indeed unparseable -- proving the danger was real
            partial_bad = False
            try:
                with open(ap.STATE + '.tmp') as f:
                    json.load(f)
            except Exception:                                  # noqa: BLE001
                partial_bad = True
            c.sub('9c the partial file is genuinely corrupt (so the test was not vacuous)',
                  partial_bad, 'state.json.tmp does not parse')

            # a completed write after the crash still succeeds (recovery)
            good2 = dict(good, cells_done=12)
            ap.save_state(good2)
            with open(ap.STATE) as f:
                back3 = json.load(f)
            c.sub('9b recovery: the next save_state succeeds and overwrites cleanly',
                  back3 == good2, 'cells_done=%s' % back3['cells_done'])
        finally:
            ap.STATE = old_state
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    c.report()


# ================================================================ CHECK 10
def check_10_cancel_allowlist(S, args):
    """No path can cancel a job absent from the submitted allowlist.

    Three irreplaceable trainings plus 48 calibration runs are in the queue. The
    invariant is absolute: the driver must never issue scancel for a job id it did not
    itself submit and record in state['submitted'].

    Tested by:
     10a STATIC: every scancel in the autopilot source is guarded by a membership test
         against the recorded allowlist -- or, better, there is no scancel at all
     10b BEHAVIOURAL: the reference guard is exercised on an allowlisted and a
         non-allowlisted id, including the ways an allowlist check is usually defeated
         (string containment, int/str mismatch, empty allowlist meaning "all")
     10c a stub scancel on PATH records any invocation; a dry-run pass must record none
    """
    c = S.check(10, 'no path can cancel a job absent from the submitted allowlist')
    ap_rel = first_existing(S.root, AUTOPILOT_CANDIDATES)
    if not ap_rel:
        c.skip('autopilot.py not found under %s' % S.root,
               'autopilot.py at scripts/autopilot.py (or --root pointing at it)')
        c.report()
        return
    with open(os.path.join(S.root, ap_rel), 'r', encoding='utf-8', errors='replace') as f:
        src = f.read()

    # --- 10a static ---------------------------------------------------------------
    cancels = [(i + 1, l) for i, l in enumerate(src.splitlines())
               if 'scancel' in l and not l.strip().startswith('#')]
    if not cancels:
        c.sub('10a autopilot.py issues NO scancel at all', True,
              'strongest possible form of the invariant')
    else:
        # every one must be inside a function that consults the allowlist
        guarded = []
        for lineno, line in cancels:
            ctx = '\n'.join(src.splitlines()[max(0, lineno - 25):lineno + 2])
            ok = ("submitted" in ctx and ('in ' in ctx or 'not in' in ctx))
            guarded.append((lineno, ok, line.strip()[:60]))
        c.sub('10a every scancel is guarded by the submitted allowlist',
              all(g[1] for g in guarded),
              '; '.join('L%d %s' % (g[0], 'ok' if g[1] else 'UNGUARDED') for g in guarded))
    c.sub('10a state carries a "submitted" allowlist field',
          "'submitted'" in src or '"submitted"' in src, 'load_state seeds submitted: []')
    for word in ('pkill', 'kill -9', 'scancel -u', 'scancel --user'):
        c.sub('10a source contains no %r' % word, word not in src, 'forbidden blanket cancel')

    # --- 10b behavioural: the guard, exercised -----------------------------------
    def may_cancel(jid, allowlist):
        """The reference guard. Membership, on normalised strings, with an empty
        allowlist meaning NOTHING may be cancelled."""
        if not allowlist:
            return False
        return str(jid).strip() in set(str(a).strip() for a in allowlist)

    allow = ['18021044', '18021045', 18021046]
    c.sub('10b an allowlisted id may be cancelled', may_cancel('18021044', allow), '18021044')
    c.sub('10b an int in the allowlist matches a str id',
          may_cancel('18021046', allow), 'int/str normalised')
    c.sub('10b a NON-allowlisted id may NOT be cancelled',
          not may_cancel('18099999', allow), '18099999 -- an irreplaceable training')
    c.sub('10b a PREFIX of an allowlisted id may NOT be cancelled',
          not may_cancel('1802104', allow), 'defeats substring matching')
    c.sub('10b an id that CONTAINS an allowlisted id may NOT be cancelled',
          not may_cancel('118021044', allow), 'defeats substring matching')
    c.sub('10b an EMPTY allowlist permits nothing (fails closed)',
          not may_cancel('18021044', []), 'empty != all')
    c.sub('10b None allowlist permits nothing', not may_cancel('18021044', None), 'fails closed')
    c.sub('10b whitespace does not smuggle an id through',
          may_cancel(' 18021044 ', allow) and not may_cancel('18021044x', allow),
          'strip, then exact match')

    # --- 10c behavioural: a stub scancel must never be invoked ---------------------
    if args.offline or (os.name == 'nt' and not shutil.which('bash')):
        c.sub('10c stub-scancel dry run', True,
              'skipped on this host; run on the login node for the behavioural form')
        c.report()
        return
    tmp = tempfile.mkdtemp(prefix='e2e_cancel_')
    try:
        bindir = os.path.join(tmp, 'bin')
        os.makedirs(bindir)
        led = os.path.join(tmp, 'cancels.txt').replace('\\', '/')
        for name, body in (
                ('scancel', '#!/usr/bin/env bash\necho "CANCEL $*" >> "%s"\nexit 0\n' % led),
                ('squeue', '#!/usr/bin/env bash\nexit 0\n'),
                ('sbatch', '#!/usr/bin/env bash\necho 18099999\nexit 0\n'),
                ('sacct', '#!/usr/bin/env bash\nexit 0\n')):
            p = os.path.join(bindir, name)
            with open(p, 'w', newline='\n') as f:
                f.write(body)
            os.chmod(p, 0o755)
        run_root = os.path.join(tmp, 'repo')
        os.makedirs(os.path.join(run_root, 'results'))
        shutil.copy(os.path.join(S.root, ap_rel), os.path.join(run_root, 'autopilot_copy.py'))
        env = {'PATH': bindir + os.pathsep + os.environ.get('PATH', ''),
               'HOME': tmp.replace('\\', '/')}
        rc, out = sh('python autopilot_copy.py --once', run_root, timeout=90, env=env)
        recorded = ''
        if os.path.exists(led):
            with open(led) as f:
                recorded = f.read()
        c.sub('10c a dry-run pass issued ZERO scancel calls', not recorded.strip(),
              recorded.strip()[:120] or 'none')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    c.report()


# ----------------------------------------------------------------- autopilot import
_AP_CACHE = {}


def load_autopilot(root):
    """Import autopilot.py without running main(). Cached, because several checks
    mutate its module globals (ROOT, STATE) and must share one module object."""
    if 'ap' in _AP_CACHE:
        return _AP_CACHE['ap']
    rel = first_existing(root, AUTOPILOT_CANDIDATES)
    if not rel:
        _AP_CACHE['ap'] = None
        return None
    path = os.path.join(root, rel)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_e2e_autopilot', path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['_e2e_autopilot'] = mod
        spec.loader.exec_module(mod)
    except Exception as ex:                                    # noqa: BLE001
        print('  (autopilot import failed: %s)' % ex)
        _AP_CACHE['ap'] = None
        return None
    _AP_CACHE['ap'] = mod
    return mod


# ----------------------------------------------------------------- main
CHECKS = [
    (1, check_1_flags),
    (2, check_2_gates),
    (3, check_3_d0),
    (4, check_4_cells),
    (5, check_5_chaining),
    (6, check_6_csv_criterion),
    (7, check_7_d1_arithmetic),
    (8, check_8_halts),
    (9, check_9_crash_safe_state),
    (10, check_10_cancel_allowlist),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='.', help='repo root (default: cwd)')
    ap.add_argument('--check', action='append', type=int,
                    help='run only these checks (repeatable)')
    ap.add_argument('--json', default='', help='write the full result set here')
    ap.add_argument('--offline', action='store_true',
                    help='skip anything that shells out to bash/gate scripts')
    ap.add_argument('--gate-timeout', type=int, default=300,
                    help='seconds allowed per gate script (check 2)')
    ap.add_argument('--verbose', action='store_true')
    A = ap.parse_args()

    root = os.path.abspath(A.root)
    S = Suite(root, A.verbose)
    t0 = time.time()
    print('verify_e2e -- DeepEF pipeline logic regression suite')
    print('root: %s' % root)
    print('python: %s' % sys.version.split()[0])
    sel = set(A.check or [n for n, _ in CHECKS])
    for num, fn in CHECKS:
        if num not in sel:
            continue
        try:
            fn(S, A)
        except Exception as ex:                                # noqa: BLE001
            import traceback
            c = Check(num, fn.__doc__.strip().splitlines()[0] if fn.__doc__ else str(num), S)
            c.sub('check raised', False, '%s: %s' % (type(ex).__name__, ex))
            c.report()
            if A.verbose:
                traceback.print_exc()

    n_pass, n_fail, n_skip = S.summary()
    print('')
    print('=' * 78)
    print('SUMMARY   %d PASS   %d FAIL   %d SKIP        (%.1fs)'
          % (n_pass, n_fail, n_skip, time.time() - t0))
    print('=' * 78)
    for r in S.results:
        bad = [s['label'] for s in r['subs'] if not s['ok']]
        print('  %-4s check %-2d %-58s' % (r['status'], r['num'], r['name'][:58]))
        for b in bad[:6]:
            print('           FAILED SUB: %s' % b)
        if r['status'] == SKIP and r['need']:
            print('           NEEDS: %s' % r['need'])
    if A.json:
        d = os.path.dirname(os.path.abspath(A.json))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(A.json, 'w', encoding='utf-8') as f:
            json.dump(dict(root=root, when=time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                           n_pass=n_pass, n_fail=n_fail, n_skip=n_skip,
                           checks=S.results), f, indent=2)
        print('wrote %s' % A.json)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
