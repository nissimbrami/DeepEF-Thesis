"""Extract structural features from AlphaFold/Rosetta PDBs for the 28 test proteins."""
import os, sys, math, json
import numpy as np, pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_index, index_to_one
from scipy.spatial import ConvexHull, cKDTree

AFD = 'data/Processed_K50_dG_datasets/AlphaFold_model_PDBs/'
parser = PDBParser(QUIET=True)

KD = dict(A=1.8,R=-4.5,N=-3.5,D=-3.5,C=2.5,Q=-3.5,E=-3.5,G=-0.4,H=-3.2,I=4.5,
          L=3.8,K=-3.9,M=1.9,F=2.8,P=-1.6,S=-0.8,T=-0.7,W=-0.9,Y=-1.3,V=4.2)
MAXASA = dict(A=129,R=274,N=195,D=193,C=167,Q=225,E=223,G=104,H=224,I=197,
              L=201,K=236,M=224,F=240,P=159,S=155,T=172,W=285,Y=263,V=174)
VDW = {'C':1.70,'N':1.55,'O':1.52,'S':1.80,'H':1.20,'P':1.80}


def sasa_shrake(coords, radii, n_pts=200, probe=1.4):
    i = np.arange(n_pts) + 0.5
    phi = np.arccos(1 - 2*i/n_pts)
    theta = np.pi*(1+5**0.5)*i
    sph = np.stack([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)], 1)
    R = radii + probe
    tree = cKDTree(coords)
    areas = np.zeros(len(coords))
    maxR = R.max()
    for k in range(len(coords)):
        nb = np.array(tree.query_ball_point(coords[k], R[k]+maxR), dtype=int)
        nb = nb[nb != k]
        pts = coords[k] + sph*R[k]
        if len(nb):
            d2 = ((pts[:,None,:]-coords[nb][None,:,:])**2).sum(-1)
            acc = (d2 >= (R[nb]**2)[None,:]).all(1)
        else:
            acc = np.ones(n_pts, bool)
        areas[k] = 4*np.pi*R[k]**2 * acc.mean()
    return areas


def dihe(p0,p1,p2,p3):
    b0=p0-p1; b1=p2-p1; b2=p3-p2
    b1n=b1/np.linalg.norm(b1)
    v=b0-np.dot(b0,b1n)*b1n; w=b2-np.dot(b2,b1n)*b1n
    x=np.dot(v,w); y=np.dot(np.cross(b1n,v),w)
    return math.degrees(math.atan2(y,x))


def phipsi_ss(chain_res):
    n = len(chain_res)
    ss = ['C']*n
    def atom(r, name):
        return r[name].get_coord() if name in r else None
    phis=[None]*n; psis=[None]*n
    for i in range(n):
        r=chain_res[i]
        N,CA,C = atom(r,'N'),atom(r,'CA'),atom(r,'C')
        if N is None or CA is None or C is None: continue
        if i>0:
            Cp=atom(chain_res[i-1],'C')
            if Cp is not None: phis[i]=dihe(Cp,N,CA,C)
        if i<n-1:
            Nn=atom(chain_res[i+1],'N')
            if Nn is not None: psis[i]=dihe(N,CA,C,Nn)
    for i in range(n):
        p,s = phis[i], psis[i]
        if p is None or s is None: continue
        if -160 <= p <= -20 and -120 <= s <= 50:
            ss[i]='H'
        elif -180 <= p <= -40 and (s >= 90 or s <= -150):
            ss[i]='E'
    out = ['C']*n
    for lab, minrun in (('H',4),('E',3)):
        i=0
        while i<n:
            if ss[i]==lab:
                j=i
                while j<n and ss[j]==lab: j+=1
                if j-i>=minrun:
                    for k in range(i,j): out[k]=lab
                i=j
            else: i+=1
    return out, phis, psis


def feats(prot):
    st = parser.get_structure(prot, AFD+prot+'.pdb')
    model = list(st)[0]
    res = [r for ch in model for r in ch if r.id[0]==' ']
    n = len(res)
    seq = []
    for r in res:
        try: seq.append(index_to_one(three_to_index(r.get_resname())))
        except Exception: seq.append('X')
    seq = ''.join(seq)

    ca = np.array([r['CA'].get_coord() for r in res if 'CA' in r])
    hv = [(a, ri) for ri, r in enumerate(res) for a in r if a.element != 'H']
    hcoords = np.array([a.get_coord() for a,_ in hv])
    hrad = np.array([VDW.get(a.element, 1.7) for a,_ in hv])
    hres = np.array([ri for _,ri in hv])

    f = {'protein': prot, 'n_res': n}
    bf = np.array([a.get_bfactor() for a,_ in hv])
    f['bfac_mean'] = float(bf.mean()); f['bfac_std'] = float(bf.std())
    f['bfac_all_zero'] = int(np.allclose(bf, 0))

    c = ca - ca.mean(0)
    f['Rg'] = float(np.sqrt((c**2).sum(1).mean()))
    f['Rg_over_N13'] = f['Rg'] / (n**(1/3))
    T = (c[:,:,None]*c[:,None,:]).mean(0)
    ev = np.sort(np.linalg.eigvalsh(T))[::-1]
    f['asphericity'] = float((ev[0] - 0.5*(ev[1]+ev[2])) / ev.sum())
    f['acylindricity'] = float((ev[1]-ev[2])/ev.sum())
    f['shape_aniso'] = float((ev**2).sum()/ev.sum()**2*1.5 - 0.5)
    try:
        hull = ConvexHull(hcoords)
        f['hull_vol'] = float(hull.volume); f['hull_area'] = float(hull.area)
        f['sphericity'] = float((np.pi**(1/3)*(6*hull.volume)**(2/3))/hull.area)
        f['vol_per_res'] = float(hull.volume/n)
    except Exception:
        f['hull_vol']=f['hull_area']=f['sphericity']=f['vol_per_res']=np.nan

    D = np.linalg.norm(ca[:,None]-ca[None,:], axis=-1)
    idx = np.arange(len(ca))
    sep = np.abs(idx[:,None]-idx[None,:])
    m = (D < 8.0) & (sep >= 3)
    ii,jj = np.where(np.triu(m,1))
    ncont = len(ii)
    f['n_contacts'] = ncont
    f['contact_density'] = ncont/n
    if ncont:
        f['CO_abs'] = float((jj-ii).mean())
        f['RCO'] = float((jj-ii).sum()/(ncont*n))
        f['LRO'] = float(((jj-ii) >= 12).sum()/n)
        f['frac_longrange'] = float(((jj-ii)>=12).mean())
    else:
        f['CO_abs']=f['RCO']=f['LRO']=f['frac_longrange']=np.nan

    tree = cKDTree(hcoords)
    prs = tree.query_pairs(6.0, output_type='ndarray')
    ri, rj = hres[prs[:,0]], hres[prs[:,1]]
    okm = np.abs(ri-rj) >= 3
    ri, rj = ri[okm], rj[okm]
    if len(ri):
        pa = np.unique(np.stack([np.minimum(ri,rj), np.maximum(ri,rj)],1), axis=0)
        f['CO_heavy_abs'] = float((pa[:,1]-pa[:,0]).mean())
        f['RCO_heavy'] = float((pa[:,1]-pa[:,0]).sum()/(len(pa)*n))
        f['heavy_contact_density'] = len(pa)/n
    else:
        f['CO_heavy_abs']=f['RCO_heavy']=f['heavy_contact_density']=np.nan

    sasa_a = sasa_shrake(hcoords, hrad)
    f['SASA_total'] = float(sasa_a.sum())
    f['SASA_per_res'] = float(sasa_a.sum()/n)
    f['SASA_over_N23'] = float(sasa_a.sum()/n**(2/3))
    sc_mask = np.array([a.get_id() not in ('N','C','O','OXT') for a,_ in hv])
    sc = np.zeros(n); np.add.at(sc, hres[sc_mask], sasa_a[sc_mask])
    rsa = np.array([sc[i]/MAXASA.get(seq[i],200) for i in range(n)])
    f['RSA_mean'] = float(rsa.mean())
    f['frac_buried_rsa20'] = float((rsa < 0.20).mean())
    f['frac_buried_rsa05'] = float((rsa < 0.05).mean())
    f['frac_exposed_rsa50'] = float((rsa > 0.50).mean())
    hyd = np.array([KD.get(a,0) for a in seq])
    f['mean_hydropathy'] = float(hyd.mean())
    f['buried_hydrophobic_frac'] = float(((rsa<0.20)&(hyd>1.5)).sum()/n)
    f['exposed_hydrophobic_frac'] = float(((rsa>0.5)&(hyd>1.5)).sum()/n)
    f['hydrophobic_moment_proxy'] = float(np.corrcoef(hyd, -rsa)[0,1]) if n>3 else np.nan

    surf = hcoords[sasa_a > 5.0]
    if len(surf) > 3:
        st_ = cKDTree(surf)
        dd,_ = st_.query(ca)
        f['depth_mean'] = float(dd.mean()); f['depth_max'] = float(dd.max())
        f['depth_p90'] = float(np.percentile(dd,90))
        f['frac_deep_6A'] = float((dd>6.0).mean())
    else:
        f['depth_mean']=f['depth_max']=f['depth_p90']=f['frac_deep_6A']=np.nan

    ssl, phis, psis = phipsi_ss(res)
    ss = ''.join(ssl)
    f['frac_H'] = ss.count('H')/n; f['frac_E'] = ss.count('E')/n; f['frac_C'] = ss.count('C')/n
    segs = [s for s in ss.replace('E','|').replace('C','|').split('|') if s]
    f['n_helix_seg'] = len(segs)
    f['mean_helix_len'] = float(np.mean([len(s) for s in segs])) if segs else 0.0
    segsE = [s for s in ss.replace('H','|').replace('C','|').split('|') if s]
    f['n_strand_seg'] = len(segsE)
    f['ss_order'] = f['frac_H']+f['frac_E']

    Ns = {}; Os = {}
    for ri_, r in enumerate(res):
        if 'N' in r: Ns[ri_]=r['N'].get_coord()
        if 'O' in r: Os[ri_]=r['O'].get_coord()
    nhb=0; nhb_lr=0
    okeys=list(Os.keys()); ocoord=np.array([Os[k] for k in okeys])
    otree=cKDTree(ocoord)
    for i_, Nc in Ns.items():
        for oi in otree.query_ball_point(Nc, 3.5):
            j_=okeys[oi]
            if abs(i_-j_) < 2: continue
            nhb+=1
            if abs(i_-j_) > 5: nhb_lr+=1
    f['n_hbond'] = nhb; f['hbond_per_res'] = nhb/n
    f['hbond_lr_per_res'] = nhb_lr/n
    f['hbond_lr_frac'] = nhb_lr/nhb if nhb else np.nan

    posA=[]; negA=[]
    for ri_, r in enumerate(res):
        rn=r.get_resname()
        if rn in ('LYS','ARG','HIS'):
            for a in r:
                if a.element=='N' and a.get_id()!='N': posA.append((a.get_coord(),ri_))
        if rn in ('ASP','GLU'):
            for a in r:
                if a.element=='O' and a.get_id() not in ('O','OXT'): negA.append((a.get_coord(),ri_))
    sbpairs=set()
    if posA and negA:
        nc=np.array([x[0] for x in negA])
        nt=cKDTree(nc)
        for pcd,pri in posA:
            for oi in nt.query_ball_point(pcd, 4.0):
                sbpairs.add((pri, negA[oi][1]))
    f['n_saltbridge'] = len(sbpairs); f['saltbridge_per_res'] = len(sbpairs)/n
    f['frac_charged'] = float(sum(1 for a in seq if a in 'KRDE')/n)
    f['net_charge'] = float(sum(1 for a in seq if a in 'KR') - sum(1 for a in seq if a in 'DE'))
    f['net_charge_per_res'] = f['net_charge']/n

    sg=[(r['SG'].get_coord(), ri_) for ri_,r in enumerate(res) if r.get_resname()=='CYS' and 'SG' in r]
    nss=0
    for a in range(len(sg)):
        for b in range(a+1,len(sg)):
            if np.linalg.norm(sg[a][0]-sg[b][0]) < 2.5: nss+=1
    f['n_disulfide']=nss; f['n_cys']=seq.count('C')

    if not np.isnan(f['hull_vol']):
        avol = (4/3)*np.pi*(hrad**3).sum()
        f['packing_frac'] = float(avol/f['hull_vol'])
        f['void_vol_per_res'] = float((f['hull_vol']-avol)/n)
    else:
        f['packing_frac']=f['void_vol_per_res']=np.nan
    nb10 = (D < 10.0).sum(1) - 1
    f['ca_nb10_mean'] = float(nb10.mean())
    f['ca_nb10_std'] = float(nb10.std())
    f['frac_core_nb10_gt20'] = float((nb10 > 20).mean())

    f['frac_gly'] = seq.count('G')/n; f['frac_pro'] = seq.count('P')/n
    f['frac_aromatic'] = sum(seq.count(a) for a in 'FWY')/n
    f['frac_ILVFM'] = sum(seq.count(a) for a in 'ILVFM')/n
    f['seq'] = seq; f['ss_string'] = ss
    return f


if __name__ == '__main__':
    prots = sys.argv[1].split(',')
    rows=[]
    for p in prots:
        try:
            rows.append(feats(p)); print('ok', p, flush=True)
        except Exception as e:
            print('FAIL', p, repr(e), flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(sys.argv[2], index=False)
    print('wrote', sys.argv[2], df.shape)
