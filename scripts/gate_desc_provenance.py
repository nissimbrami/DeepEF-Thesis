"""GATE: the descriptor-matrix naming trap is closed.

Asserts, by RUNNING it:
  * every live mode loads the file its name states, and only that file;
  * the retired names (curated12 / pca16 / pca16_only) RAISE rather than aliasing;
  * the provenance assertion FIRES on right-name/wrong-content, on a stripped header,
    and on a wrong column count -- the three ways the curated12 trap can recur;
  * an explicit --aa_descriptor_csv override relaxes the header check but still
    enforces K;
  * descriptor_provenance() returns a distinct md5 per matrix.

Every negative case is checked for a RAISE. A guard that silently returns is the
project's signature failure mode, so "no raise" is a FAILURE here, never a pass.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.getcwd())
os.environ.setdefault('WANDB_MODE', 'disabled')

import aa_descriptors as ad

ok = True
def check(name, cond, extra=''):
    global ok
    print('%-64s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))
    ok = ok and bool(cond)


def fresh():
    """Drop the table cache so the next load really touches disk."""
    ad._CACHE.clear()


def expect_raise(name, fn, want_substr=None):
    """The negative-test primitive. NO RAISE is a FAILURE."""
    fresh()
    try:
        fn()
    except (ValueError, KeyError, FileNotFoundError) as e:
        msg = str(e)
        if want_substr and want_substr not in msg:
            check(name, False, 'raised, but message lacks %r' % want_substr)
            return msg
        check(name, True, '-> %s' % msg.splitlines()[0][:72])
        return msg
    check(name, False, '*** RETURNED WITHOUT RAISING -- the guard is dead ***')
    return None


print('=== live modes resolve to correctly-named files ===')
EXPECT = {
    'mordred726': ('data/aa_descriptors_mordred.csv', 726),
    'mordred_pca16': ('data/aa_descriptors_mordred_pca16.csv', 16),
    'mordred_pca16_only': ('data/aa_descriptors_mordred_pca16.csv', 16),
}
check('MODES is exactly the renamed set',
      ad.MODES == ('none', 'mordred726', 'mordred_pca16', 'mordred_pca16_only'),
      str(ad.MODES))
for m, (want_path, want_k) in sorted(EXPECT.items()):
    fresh()
    check('%s -> %s' % (m, want_path), ad.descriptor_path(m) == want_path,
          ad.descriptor_path(m))
    check('%s has K=%d' % (m, want_k), ad.descriptor_dim(m) == want_k,
          'K=%d' % ad.descriptor_dim(m))
    # the name must contain the pipeline word, so a mismatch is visible in the path
    check('%s filename names its pipeline' % m, 'mordred' in want_path, want_path)

print()
print('=== retired names RAISE (not aliased) ===')
for m in ('curated12', 'pca16', 'pca16_only'):
    expect_raise('retired mode %r raises' % m,
                 lambda m=m: ad.descriptor_dim(m), 'RETIRED')
expect_raise('unknown mode raises', lambda: ad.descriptor_dim('banana'))

print()
print('=== the provenance assertion FIRES ===')
tmp = tempfile.mkdtemp(prefix='descprov_')
saved = dict(ad._DEFAULT_FILES)


def point(mode, path):
    ad._DEFAULT_FILES[mode] = os.path.abspath(path)


def restore():
    ad._DEFAULT_FILES.clear()
    ad._DEFAULT_FILES.update(saved)
    fresh()


# (1) RIGHT NAME, WRONG CONTENT -- the exact curated12 trap. A file that is really the
# 726-col full matrix is placed where the 16-col PCA arm expects its matrix. Note the
# temp file is given a NON-colliding basename: _resolve() tries several roots, so a temp
# file named like the real one could be shadowed by the real one and the test would
# vacuously pass.
bad_content = os.path.join(tmp, 'trap_full_matrix_posing_as_pca.csv')
shutil.copy('data/aa_descriptors_mordred.csv', bad_content)
point('mordred_pca16', bad_content)
check('setup: trap file really is the 726-col matrix',
      ad.descriptor_path('mordred_pca16') == bad_content, bad_content)
msg = expect_raise('right name / wrong content raises',
                   lambda: ad.descriptor_dim('mordred_pca16'),
                   'DESCRIPTOR PROVENANCE MISMATCH')
if msg:
    check('  error states what was EXPECTED', 'expected' in msg.lower())
    check('  error states what was FOUND', 'found' in msg.lower())
    check('  error names the offending file', bad_content in msg)
    check('  error reports the real column count', '726' in msg)
restore()

# (2) HEADER STRIPPED -- an untraceable file must not load.
noheader = os.path.join(tmp, 'trap_no_provenance_header.csv')
with open('data/aa_descriptors_mordred_pca16.csv') as fi, open(noheader, 'w') as fo:
    for ln in fi:
        if not ln.startswith('#'):
            fo.write(ln)
point('mordred_pca16', noheader)
msg = expect_raise('stripped provenance header raises',
                   lambda: ad.descriptor_dim('mordred_pca16'),
                   'DESCRIPTOR PROVENANCE MISMATCH')
if msg:
    check('  error says the header is missing', 'NO "#" provenance header' in msg)
restore()

# (3) THE OLD CURATED PLACEHOLDER under a mordred mode -- wrong K AND wrong header.
point('mordred726', 'data/aa_descriptors.PLACEHOLDER.bak.csv')
msg = expect_raise('old curated placeholder under mordred726 raises',
                   lambda: ad.descriptor_dim('mordred726'),
                   'DESCRIPTOR PROVENANCE MISMATCH')
if msg:
    check('  error reports the 14-col width', '14' in msg)
    check('  error flags the curated header', 'curated literature table' in msg)
restore()

# (4) open25 (25 residues, 756 cols) under mordred726 -- right pipeline, wrong matrix.
point('mordred726', 'data/aa_descriptors_open25.csv')
# NOTE: this one is caught EARLIER by the pre-existing W10 column-identity guard,
# which refuses a table whose column NAMES differ from the canonical matrix. That is a
# strictly better error than the provenance mismatch would be, so accept either -- the
# requirement is that it REFUSES, not which of the two guards gets there first.
msg = expect_raise('open25 matrix under mordred726 raises (756 != 726)',
                   lambda: ad.descriptor_dim('mordred726'))
if msg:
    check('  error reports 756 found vs 726 expected',
          '756' in msg and '726' in msg)
    check('  refusal comes from a named guard (W10 columns or provenance)',
          ('W10' in msg) or ('DESCRIPTOR PROVENANCE MISMATCH' in msg),
          msg.splitlines()[0][:60])
restore()

print()
print('=== --aa_descriptor_csv override: header relaxed, K STILL enforced ===')


class Cfg(object):
    def __init__(self, csv):
        self.aa_descriptor_csv = csv
        self.aa_descriptors = 'mordred_pca16'


# Same as above: the W10 column-identity guard may fire first. Either refusal is fine.
m_ov = expect_raise('override with wrong K still raises',
                    lambda: ad.descriptor_dim('mordred_pca16',
                                              Cfg('data/aa_descriptors_open25.csv')))
if m_ov:
    check('  override refusal is from a named guard',
          ('W10' in m_ov) or ('DESCRIPTOR PROVENANCE MISMATCH' in m_ov),
          m_ov.splitlines()[0][:60])

fresh()
try:
    # open25 has 756 cols; mordred726 expects 726 -> must still raise even under override
    ad.descriptor_dim('mordred726', Cfg('data/aa_descriptors_open25.csv'))
    check('override with wrong K on mordred726 raises', False,
          '*** RETURNED WITHOUT RAISING ***')
except ValueError:
    check('override with wrong K on mordred726 raises', True)

fresh()
# An override whose K MATCHES the mode is allowed through (header relaxed by design).
k = ad.descriptor_dim('mordred_pca16',
                      Cfg('data/aa_descriptors_mordred_pca16.csv'))
check('override with matching K is allowed', k == 16, 'K=%d' % k)

print()
print('=== descriptor_provenance() is a real trace ===')
fresh()
recs = dict((m, ad.descriptor_provenance(m)) for m in ad.MODES)
check('mode none has no csv/md5', recs['none']['aa_desc_md5'] is None)
md5_full = recs['mordred726']['aa_desc_md5']
md5_pca = recs['mordred_pca16']['aa_desc_md5']
check('md5 recorded for mordred726', bool(md5_full) and len(md5_full) == 32, md5_full)
check('md5 recorded for mordred_pca16', bool(md5_pca) and len(md5_pca) == 32, md5_pca)
check('the two matrices have DIFFERENT md5', md5_full != md5_pca)
check('pca16 and pca16_only share one matrix (same md5)',
      md5_pca == recs['mordred_pca16_only']['aa_desc_md5'])
for m in ('mordred726', 'mordred_pca16'):
    r = recs[m]
    check('%s provenance carries generator+shape' % m,
          r['aa_desc_provenance'] and 'generator=' in r['aa_desc_provenance']
          and 'final_shape=' in r['aa_desc_provenance'])
    check('%s csv path is absolute+resolved' % m, os.path.isabs(r['aa_desc_csv']),
          r['aa_desc_csv'])

# md5 must equal the real file digest -- not something invented.
import hashlib
real = hashlib.md5(open(ad.descriptor_path('mordred726'), 'rb').read()).hexdigest()
check('recorded md5 equals the real file digest', real == md5_full, real)

shutil.rmtree(tmp, ignore_errors=True)
restore()
print()
print('DESC-PROVENANCE GATE: %s' % ('ALL PASS' if ok else 'FAILURES'))
sys.exit(0 if ok else 1)
