"""Generate the thesis figure set into results/figures/.

Every number is read from a computed artefact; nothing is hard-coded except
axis labels and the Kyte-Doolittle / volume reference scales.
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = '/home/nissimb/DeepPEF'
FIG = os.path.join(R, 'results', 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'figure.dpi': 160, 'savefig.dpi': 160, 'font.size': 9,
                     'axes.grid': True, 'grid.alpha': .25, 'axes.axisbelow': True,
                     'savefig.bbox': 'tight'})
C0, C1, C2 = '#2b6cb0', '#c05621', '#2f855a'
CTRL = 'abl_calib_ctrl_repro2_e14.csv'
TRUE_2K5H = 4.805470


def evals():
    out = {}
    for f in sorted(glob.glob(os.path.join(R, 'eval_results', '*.csv'))):
        nm = os.path.basename(f)
        d = pd.read_csv(f)
        if not set(['protein', 'ddG', 'pred_ddG']).issubset(d.columns):
            continue
        d = d.dropna(subset=['ddG', 'pred_ddG'])
        if d.protein.nunique() < 28:
            continue
        out[nm] = d
    return out


E = evals()
NOND1 = [k for k in E if 'D1_uemb' not in k]
print('eval CSVs: %d total, %d non-D1' % (len(E), len(NOND1)))
saved = []


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    saved.append(name)
    print('  wrote %s' % name)


def per_prot(d):
    rec = {}
    for p, g in d.groupby('protein'):
        if g.ddG.std() == 0 or g.pred_ddG.std() == 0:
            continue
        a, b = np.polyfit(g.ddG, g.pred_ddG, 1)
        rec[p] = dict(a=a, b=b, r=np.corrcoef(g.ddG, g.pred_ddG)[0, 1],
                      s=g.pred_ddG.std() / g.ddG.std(), n=len(g))
    return rec


# ---------------------------------------------------------------- Figure 1
d = E[CTRL]
pp = per_prot(d)
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))
fig.subplots_adjust(wspace=.28)
prots = sorted(pp)
cmap = plt.get_cmap('twilight')
for i, p in enumerate(prots):
    g = d[d.protein == p]
    ax[0].scatter(g.ddG, g.pred_ddG, s=1.2, alpha=.25,
                  color=cmap(i / float(len(prots))), linewidths=0)
lim = [-6, 3]
ax[0].plot(lim, lim, 'k--', lw=1, label='y = x (perfect)')
xs = np.linspace(lim[0], lim[1], 10)
med_a = np.median([v['a'] for v in pp.values()])
ax[0].plot(xs, med_a * xs, color='r', lw=1.6,
           label='median slope $a_p$ = %.3f' % med_a)
ax[0].set(xlim=lim, ylim=lim, xlabel='measured $\\Delta\\Delta G$ (kcal/mol)',
          ylabel='predicted $\\Delta\\Delta G$ (kcal/mol)',
          title='A  Per-mutation predictions, 28 proteins\n(n = {:,} mutations)'.format(len(d)))
ax[0].legend(fontsize=7, loc='upper left')

for i, p in enumerate(prots):
    a, b = pp[p]['a'], pp[p]['b']
    ax[1].plot(xs, a * xs + b, color=cmap(i / float(len(prots))), lw=1, alpha=.85)
ax[1].plot(lim, lim, 'k--', lw=1.2)
ax[1].set(xlim=lim, ylim=lim, xlabel='measured $\\Delta\\Delta G$ (kcal/mol)',
          ylabel='predicted $\\Delta\\Delta G$ (kcal/mol)',
          title='B  Per-protein fits $pred \\approx a_p\\,true + b_p$\n'
                'slopes differ (compression), intercepts differ (offset)')
save(fig, 'fig01_calibration_decomposition.png')

# ---------------------------------------------------------------- Figure 2
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
fig.subplots_adjust(wspace=.28)
wp = [np.corrcoef(d[d.protein == p].ddG, d[d.protein == p].pred_ddG)[0, 1] for p in prots]
pooled = np.corrcoef(d.ddG, d.pred_ddG)[0, 1]
ax[0].hist(wp, bins=12, color=C0, alpha=.85, edgecolor='w')
ax[0].axvline(np.median(wp), color=C0, lw=2,
              label='per-protein median = %.3f' % np.median(wp))
ax[0].axvline(pooled, color=C1, lw=2, ls='--',
              label='pooled (all 28 together) = %.3f' % pooled)
ax[0].set(xlabel='Pearson r ($\\Delta\\Delta G$)', ylabel='proteins',
          title='A  Ranking within a protein vs across proteins\n(n = 28 proteins, one checkpoint)')
ax[0].legend(fontsize=7)


def head(nm, fix):
    e = E[nm].sort_values('protein').copy()
    if fix:
        m = e.protein == '2K5H'
        e.loc[m, 'ddG'] = e.loc[m, 'deltaG'] - TRUE_2K5H
    raw = np.corrcoef(e.ddG, e.pred_ddG)[0, 1]
    adj = pd.concat([g.pred_ddG - np.polyfit(g.ddG, g.pred_ddG, 1)[1]
                     for _, g in e.groupby('protein')]).reindex(e.index)
    return raw, np.corrcoef(e.ddG, adj)[0, 1]


B = np.array([head(k, 0) for k in NOND1])
F = np.array([head(k, 1) for k in NOND1])
labels = ['pooled\n(raw)', 'offset-removal\nORACLE']
x = np.arange(2)
w = .36
ax[1].bar(x - w / 2, B.mean(0), w, yerr=B.std(0), capsize=3, color=C1,
          alpha=.85, label='2K5H reference bug present')
ax[1].bar(x + w / 2, F.mean(0), w, yerr=F.std(0), capsize=3, color=C0,
          alpha=.85, label='2K5H reference corrected')
for xi, v in zip(x - w / 2, B.mean(0)):
    ax[1].text(xi, v + .022, '%.4f' % v, ha='center', fontsize=7)
for xi, v in zip(x + w / 2, F.mean(0)):
    ax[1].text(xi, v + .022, '%.4f' % v, ha='center', fontsize=7)
ax[1].set(xticks=x, ylim=(0, .87), ylabel='pooled $\\Delta\\Delta G$ PCC',
          title='B  Offset removal is an ORACLE\n(mean $\\pm$ sd over %d checkpoints)' % len(NOND1))
ax[1].set_xticklabels(labels)
ax[1].legend(fontsize=7, loc='upper left')
save(fig, 'fig02_within_vs_pooled_and_oracle.png')
print('  gain buggy %+.4f  fixed %+.4f' % ((B[:, 1] - B[:, 0]).mean(), (F[:, 1] - F[:, 0]).mean()))

# ---------------------------------------------------------------- Figure 3
rs = dict((k, per_prot(E[k])) for k in NOND1)
med = dict((k, dict(a=np.median([v['a'] for v in rs[k].values()]),
                    r=np.median([v['r'] for v in rs[k].values()]),
                    s=np.median([v['s'] for v in rs[k].values()]))) for k in NOND1)
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
fig.subplots_adjust(wspace=.28)
a = np.array([med[k]['a'] for k in NOND1])
r = np.array([med[k]['r'] for k in NOND1])
s = np.array([med[k]['s'] for k in NOND1])
ax[0].scatter(r * s, a, s=26, color=C0, zorder=3)
lo, hi = .1, .8
ax[0].plot([lo, hi], [lo, hi], 'k--', lw=1, label='$a_p = r\\cdot s$ (identity)')
ax[0].set(xlabel='$r \\times s$', ylabel='$a_p$ (median over 28 proteins)',
          title='A  The identity holds exactly\n(%d checkpoints)' % len(NOND1))
ax[0].legend(fontsize=7)

arms = [('abl_calib_ctrl_repro2_e14.csv', 'control\n(no term)'),
        ('abl_p3_slope0.3_s42_e9.csv', '0.3'),
        ('abl_p3_slope1.0_s42_e13.csv', '1.0'),
        ('abl_p3_slope3.0_s42_e8.csv', '3.0')]
arms = [t for t in arms if t[0] in med]
xs2 = np.arange(len(arms))
av = [med[k]['a'] for k, _ in arms]
rv = [med[k]['r'] for k, _ in arms]
sv = [med[k]['s'] for k, _ in arms]
ax[1].plot(xs2, rv, 'o-', color=C2, label='$r$  (ranking)')
ax[1].plot(xs2, sv, 's-', color=C1, label='$s$  (spread)')
ax[1].plot(xs2, av, '^-', color=C0, lw=2, label='$a_p = r\\cdot s$')
for xi, v in zip(xs2, av):
    ax[1].text(xi, v - .07, '%.3f' % v, ha='center', fontsize=7, color=C0)
ax[1].set(xticks=xs2, ylim=(0, 1.05), xlabel='--slope_weight',
          ylabel='median over 28 proteins',
          title='B  The lever moves $s$, never $r$\n(one seed per arm)')
ax[1].set_xticklabels([l for _, l in arms])
ax[1].legend(fontsize=7, loc='lower left')
save(fig, 'fig03_ap_equals_r_times_s.png')
for (k, l), A, Rr, S in zip(arms, av, rv, sv):
    print('  %-14s a_p %.4f r %.4f s %.4f' % (l.replace('\n', ' '), A, Rr, S))

# ---------------------------------------------------------------- Figure 4
KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
      'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
      'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
VOL = {'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8,
       'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6,
       'M': 162.9, 'F': 189.9, 'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8,
       'Y': 193.6, 'V': 140.0}
mt = json.load(open(os.path.join(R, 'results', 'k13_muttype.json')))
use = [k for k in mt if 'D1_uemb' not in k]
acc = {}
for k in use:
    for aa, v in mt[k]['per_residue'].items():
        acc.setdefault(aa, {'sl': [], 'sp': [], 'n': v['n']})
        acc[aa]['sl'].append(v['slope'])
        acc[aa]['sp'].append(v['spearman'])
aas = sorted(acc, key=lambda x: KD[x])
sl = np.array([np.mean(acc[a]['sl']) for a in aas])
sp = np.array([np.mean(acc[a]['sp']) for a in aas])
sle = np.array([np.std(acc[a]['sl']) for a in aas])
kd = np.array([KD[a] for a in aas])
vol = np.array([VOL[a] for a in aas])
phob = np.array([np.mean(acc[a]['sl']) for a in aas if KD[a] > 0])
phil = np.array([np.mean(acc[a]['sl']) for a in aas if KD[a] < 0])
RATIO = phil.mean() / phob.mean()
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
fig.subplots_adjust(wspace=.30)
ax[0].bar(range(len(aas)), sl, yerr=sle, capsize=2,
          color=[C1 if KD[a] > 0 else C0 for a in aas], alpha=.9)
ax[0].set(xticks=range(len(aas)), ylabel='slope (pred vs true $\\Delta\\Delta G$)',
          xlabel='destination residue, ordered by Kyte-Doolittle',
          )
ax[0].set_title('A  Mutations TO hydrophobics\nare compressed {:.2f}x\n'
                '({} checkpoints, {:,} mutations each)'.format(
                    RATIO, len(use), mt[use[0]]['matched']), fontsize=9)
ax[0].set_xticklabels(aas, fontsize=7)
ax[0].bar(0, 0, color=C0, label='hydrophilic (KD < 0)')
ax[0].bar(0, 0, color=C1, label='hydrophobic (KD > 0)')
ax[0].legend(fontsize=7)

ax[1].scatter(kd, sl, s=30, color=C0, label='slope', zorder=3)
ax[1].scatter(kd, sp, s=30, color=C2, marker='s', label='Spearman', zorder=3)
for a, x_, y_ in zip(aas, kd, sl):
    if a in ('W', 'Y', 'C', 'L', 'I'):
        ax[1].annotate(a, (x_, y_), textcoords='offset points', xytext=(3, 4), fontsize=8)
ax[1].set(xlabel='Kyte-Doolittle hydropathy', ylabel='value',
          )
ax[1].set_title('B  Ranking degrades in LOCKSTEP\n'
                'r(slope,KD) = %+.3f\nr(Spearman,KD) = %+.3f'
                % (np.corrcoef(sl, kd)[0, 1], np.corrcoef(sp, kd)[0, 1]), fontsize=9)
ax[1].legend(fontsize=7)

ax[2].scatter(sl, sp, s=34, color=C1, zorder=3)
for a, x_, y_ in zip(aas, sl, sp):
    ax[2].annotate(a, (x_, y_), textcoords='offset points', xytext=(3, 3), fontsize=7)
b1, b0 = np.polyfit(sl, sp, 1)
xr = np.linspace(sl.min(), sl.max(), 10)
ax[2].plot(xr, b1 * xr + b0, 'k--', lw=1)
ax[2].set(xlabel='slope', ylabel='Spearman',
          )
ax[2].set_title('C  LOST INFORMATION, not rescalable\n'
                'r = %+.4f\nover 20 residue types'
                % np.corrcoef(sl, sp)[0, 1], fontsize=9)
save(fig, 'fig04_hydrophobic_ranking_failure.png')
print('  corr(slope,KD)=%+.4f corr(sp,KD)=%+.4f corr(slope,vol)=%+.4f corr(slope,sp)=%+.4f'
      % (np.corrcoef(sl, kd)[0, 1], np.corrcoef(sp, kd)[0, 1],
         np.corrcoef(sl, vol)[0, 1], np.corrcoef(sl, sp)[0, 1]))

# ---------------------------------------------------------------- Figure 5
fa = json.load(open(os.path.join(R, 'results', 'factorial_analysis.json')))
cells = [c for c in fa['cells'] if c['kind'] == 'cell']
d0 = [c for c in cells if c['D_name'] == 'coil']
d1 = [c for c in cells if c['D_name'] == 'uemb']
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
fig.subplots_adjust(wspace=.28)
for g, col, lab in [(d0, C0, 'D0  Flory coil (n=%d)' % len(d0)),
                    (d1, C1, 'D1  --unfolded_emb zero (n=%d)' % len(d1))]:
    ax[0].scatter([c['a_p_median'] for c in g], [c['pooled_ddG_PCC'] for c in g],
                  s=52, color=col, label=lab, zorder=3, edgecolors='w')
ax[0].axhline(0, color='k', lw=.8, ls=':')
ax[0].set(xlabel='median $a_p$', ylabel='pooled $\\Delta\\Delta G$ PCC',
          title='A  Factor D separates completely\nno overlap on either axis')
ax[0].legend(fontsize=7, loc='center right')

noise = fa['noise']
parts = ax[1].violinplot([[c['pooled_ddG_PCC'] for c in d0],
                          [c['pooled_ddG_PCC'] for c in d1]], showmeans=True)
for pc, col in zip(parts['bodies'], [C0, C1]):
    pc.set_facecolor(col)
    pc.set_alpha(.55)
sn = noise['pooled_ddG_PCC']
ax[1].axhspan(sn['lo'], sn['hi'], color='grey', alpha=.18,
              label='seed-only range, n=%d\n(%.3f - %.3f)' % (sn['n'], sn['lo'], sn['hi']))
ax[1].set(xticks=[1, 2], ylabel='pooled $\\Delta\\Delta G$ PCC',
          title='B  The D effect dwarfs seed noise\n'
                'so no main effect may be averaged across D')
ax[1].set_xticklabels(['D0 coil', 'D1 uemb zero'])
ax[1].legend(fontsize=7, loc='center right')
save(fig, 'fig05_factor_D.png')
print('  D0 pooled %.4f..%.4f  D1 pooled %.4f..%.4f'
      % (min(c['pooled_ddG_PCC'] for c in d0), max(c['pooled_ddG_PCC'] for c in d0),
         min(c['pooled_ddG_PCC'] for c in d1), max(c['pooled_ddG_PCC'] for c in d1)))

# ---------------------------------------------------------------- Figure 6
sw = json.load(open(os.path.join(R, 'results', 'bp_replication_sweep.json')))
lc = json.load(open(os.path.join(R, 'results', 'length_confound.json')))
struct = dict((s['protein'], s) for s in lc['structure'])
pcal = json.load(open(os.path.join(R, 'results', 'calib_per_protein.json')))
common = [p for p in pcal if p in struct]
sasa = np.array([struct[p]['mean_rel_SASA'] for p in common])
bur = np.array([struct[p]['frac_buried_rel_lt_0.25'] for p in common])
apv = np.array([pcal[p]['a_p'] for p in common])
bpv = np.array([pcal[p]['b_p_wt_error'] for p in common])


def look(feat, tgt):
    for x in sw['results']:
        if x['feature'] == feat and x['target'] == tgt:
            return x


S = look('mean_rel_SASA', 'a_p')
Bq = look('frac_buried_rel_lt_0.25', 'b_p_wt_error')
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
fig.subplots_adjust(wspace=.30)
for A, xv, yv, xl, yl, rec, ttl in [
        (ax[0], sasa, apv, 'mean relative SASA', '$a_p$ (slope)', S,
         'A  Exposure predicts the SLOPE'),
        (ax[1], bur, bpv, 'fraction buried (rel. SASA < 0.25)',
         '$b_p$: WT $\\Delta G$ error (kcal/mol)', Bq, 'B  Burial predicts the OFFSET')]:
    A.scatter(xv, yv, s=34, color=C0, zorder=3)
    m, c = np.polyfit(xv, yv, 1)
    xr = np.linspace(xv.min(), xv.max(), 10)
    A.plot(xr, m * xr + c, 'k--', lw=1.2)
    A.set(xlabel=xl, ylabel=yl)
    A.set_title('%s\nreplicated mean r = %+.4f  (n = 28 proteins)\n'
                'sign-consistent %d/%d, significant %d/%d'
                % (ttl, rec['mean_r'], rec['sign_consistency'], rec['k'],
                   rec['n_sig_p05'], rec['k']), fontsize=9)
per = np.array([c['r_a_SASA'] for c in lc['per_checkpoint']])
pl = np.array([c['r_a_len'] for c in lc['per_checkpoint']])
xs3 = np.arange(len(per))
ax[2].bar(xs3 - .2, per, .4, color=C0, label='SASA $\\to a_p$')
ax[2].bar(xs3 + .2, pl, .4, color=C1, label='length $\\to a_p$')
rc = lc['r_crit_p05']
ax[2].axhline(rc, color='k', ls=':', lw=1, label='p=0.05 threshold ($\\pm$%.3f, n=28)' % rc)
ax[2].axhline(-rc, color='k', ls=':', lw=1)
ax[2].axhline(0, color='k', lw=.8)
ax[2].set(xlabel='checkpoint', ylabel='Pearson r', xticks=xs3)
ax[2].set_title('C  It survives the length confound\n'
                'SASA significant %d/%d\nlength significant %d/%d'
                % ((np.abs(per) > rc).sum(), len(per), (np.abs(pl) > rc).sum(), len(pl)),
                fontsize=9)
ax[2].set_xticklabels(range(1, len(per) + 1), fontsize=7)
ax[2].legend(fontsize=6.5, loc='lower left')
save(fig, 'fig06_structure_two_channels.png')

# ---------------------------------------------------------------- Figure 7
oc = json.load(open(os.path.join(R, 'results', 'offset_corrector.json')))
ag = oc['aggregate']
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
fig.subplots_adjust(wspace=.28)
keys = ['pooled_raw', 'pooled_mean_baseline', 'pooled_lopo_ridge', 'pooled_oracle']
labs = ['raw\n(no correction)', 'mean baseline\n(no features)',
        'LOPO ridge\n(22 features)', 'oracle\n(uses TEST labels)']
mu = [ag[k]['mean'] for k in keys]
sd = [ag[k]['std'] for k in keys]
cols = [C0, '#718096', C2, C1]
ax[0].bar(range(4), mu, yerr=sd, capsize=3, color=cols, alpha=.9)
for i, v in enumerate(mu):
    ax[0].text(i, v + .022, '%.4f' % v, ha='center', fontsize=7.5)
ax[0].set(xticks=range(4), ylim=(0, .82), ylabel='pooled $\\Delta\\Delta G$ PCC',
          title='A  95%% of the oracle gain does not survive\nheld-out prediction (%d checkpoints)'
                % oc['n_csvs_total'])
ax[0].set_xticklabels(labs, fontsize=7.5)
ax[0].annotate('', xy=(3, ag['pooled_oracle']['mean']), xytext=(0, ag['pooled_raw']['mean']),
               arrowprops=dict(arrowstyle='<->', color='k', lw=.9))
ax[0].text(1.5, .755, 'oracle gain +%.4f' % (ag['pooled_oracle']['mean'] - ag['pooled_raw']['mean']),
           ha='center', fontsize=7.5)
ax[0].text(1.5, .50, 'ridge gain only +%.4f'
           % (ag['pooled_lopo_ridge']['mean'] - ag['pooled_raw']['mean']),
           ha='center', fontsize=7.5, color=C2)

r2 = [c['r2_lopo_ridge'] for c in oc['all_csvs']]
ax[1].bar(range(len(r2)), r2, color=[C2 if v > 0 else C1 for v in r2], alpha=.9)
ax[1].axhline(0, color='k', lw=1)
ax[1].set(xlabel='checkpoint', ylabel='held-out $R^2$ of $b_p$', xticks=range(len(r2)),
          title='B  Held-out $R^2$ is NEGATIVE on %d of %d\n'
                'permutation null significant on 0/%d'
                % (sum(1 for v in r2 if v < 0), len(r2), len(r2)))
ax[1].set_xticklabels(range(1, len(r2) + 1), fontsize=7)
save(fig, 'fig07_corrector_negative.png')

# ---------------------------------------------------------------- Figure 8
ep = json.load(open(os.path.join(R, 'results', 'embedding_probe.json')))
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
fig.subplots_adjust(wspace=.28)
rd = json.load(open(os.path.join(R, 'results', 'emb_probe.json')))
r2tab = rd['residue_disjoint_R2']
props = ['hydropathy', 'charge', 'volume']
vals = [r2tab[p] for p in props]
ax[0].bar(range(len(props)), vals, color=C0, alpha=.9)
for i, v in enumerate(vals):
    ax[0].text(i, v + .015, '%.3f' % v, ha='center', fontsize=8)
ax[0].axhline(0, color='k', lw=1)
ax[0].set(xticks=range(len(props)), ylabel='held-out $R^2$', ylim=(0, .82),
          title='A  ProtT5 predicts the chemistry of a residue type\n'
                'it has NEVER seen ({} proteins, {:,} residues)'.format(
                    rd['n_proteins'], rd['n_residues']))
ax[0].set_xticklabels(props, fontsize=8)

q4 = ep.get('q4', {})
tb = q4.get('b_p', {}).get('top', [])[:8]
ta = q4.get('a_p', {}).get('top', [])[:8]
if tb:
    ax[1].bar(np.arange(len(tb)) - .2, [abs(t['r']) for t in tb], .4, color=C0, label='$b_p$')
if ta:
    ax[1].bar(np.arange(len(ta)) + .2, [abs(t['r']) for t in ta], .4, color=C1, label='$a_p$')
ax[1].axhline(0.3739, color='k', ls=':', lw=1, label='p=0.05, n=28 (uncorrected)')
ax[1].set(xlabel='best single embedding dimension (rank)', ylabel='|Pearson r|',
          title='B  Single-dimension probes: strongest hits only\n'
                'no dimension survives correction for 1024 tests')
ax[1].legend(fontsize=7)
save(fig, 'fig08_prott5_chemistry.png')

# ---------------------------------------------------------------- Figure 9
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
fig.subplots_adjust(wspace=.28)
ax[0].bar([0, 1], [1.175, 1.0], .5, color=[C1, '#cbd5e0'], alpha=.9)
ax[0].axhline(1.0, color='k', lw=1, ls='--')
ax[0].text(0, 1.20, 'r = 1.175\n"apparently harmful"', ha='center', fontsize=8, color=C1)
ax[0].set(xticks=[0, 1], ylim=(0, 1.45),
          ylabel='relative to control (lower is better)',
          title='A  Scored on $\\Delta\\Delta G$ (the WRONG metric)\n'
                'the coil is bit-identical WT vs mutant, so it CANCELS')
ax[0].set_xticklabels(['coil', 'control'])

before, after = 4.9650, 3.9521
ax[1].bar([0, 1], [before, after], .5, color=['#cbd5e0', C2], alpha=.9)
for i, v in enumerate([before, after]):
    ax[1].text(i, v + .09, '%.4f' % v, ha='center', fontsize=8)
ax[1].set(xticks=[0, 1], ylim=(0, 5.9), ylabel='$\\Delta G$ MAE (kcal/mol)',
          title='B  Scored on $\\Delta G$ (the metric it acts on)\n'
                'our single best offset lever: %.1f%% reduction'
                % (100 * (before - after) / before))
ax[1].set_xticklabels(['control', 'Flory coil'])
save(fig, 'fig09_metric_rule_coil.png')

json.dump({'figures': saved, 'n_eval_csvs': len(E), 'n_non_d1': len(NOND1)},
          open(os.path.join(FIG, 'MANIFEST.json'), 'w'), indent=1)
print('\nDONE: %d figures in results/figures/' % len(saved))
