#!/usr/bin/env python3
"""Annotate our 21 PDB-coded test proteins DIRECTLY from the PDB.

WHY THIS IS THE ONLY ROUTE. The 100k catalogue holds 67,856 complexes and full oligomeric-state
coverage, but NONE of our 28 test proteins and NONE of 247 PDB-like training proteins appear in
it: catalogue median length 477aa (p5=94) versus our 42-72aa. Two different populations.
Its Ligands and BSA columns are populated in 4 and 5 rows of 100,246 -- empty even for itself.

So the ligand / metal / complex question cannot be answered by joining. It can only be answered
by fetching our own structures. 21 of our 28 have real PDB IDs; the other 7 are designed
sequences (HHH_rd1_0244, r12_757_TrROS_Hall, ...) that have no PDB entry by construction.

WHAT IS FETCHED (RCSB REST, no key needed):
  oligomeric state, chain count, entity composition   -> is it a complex?
  non-polymer entities (HETATM ligands), excluding water and crystallisation additives
  bound metals
  experimental method and resolution

The additive filter matters: of 205,648 het-code occurrences in the catalogue only 6.6% are real
cofactors; 27.4% are crystallisation additives (SO4, GOL, EDO) and 6.3% are modified residues
like MSE, which is an amino acid IN the chain, not a ligand.
"""
import json, sys, time, urllib.request

ADDITIVES = {'HOH','SO4','GOL','EDO','PEG','MPD','DMS','ACT','CL','NA','K','FMT','TRS','EPE',
             'IPA','MES','PO4','CIT','TLA','NO3','BME','DTT','IMD','ACY','1PE','PGE','P6G'}
METALS = {'ZN','FE','MG','CA','MN','CU','NI','CO','CD','HG','NA','K','MO','W','V','SE'}

def get(url):
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None

def annotate(pdb):
    e = get(f'https://data.rcsb.org/rest/v1/core/entry/{pdb}')
    if e is None:
        return {'pdb': pdb, 'status': 'NOT FOUND'}
    out = {'pdb': pdb, 'status': 'ok'}
    out['method'] = (e.get('exptl') or [{}])[0].get('method', '?')
    out['resolution'] = (e.get('rcsb_entry_info') or {}).get('resolution_combined', [None])[0]
    ei = e.get('rcsb_entry_info') or {}
    out['polymer_entities'] = ei.get('polymer_entity_count', 0)
    out['chains'] = ei.get('deposited_polymer_entity_instance_count', 0)
    out['nonpolymer_entities'] = ei.get('nonpolymer_entity_count', 0)
    # assembly = the biologically relevant oligomeric state, not the crystal contents
    a = get(f'https://data.rcsb.org/rest/v1/core/assembly/{pdb}/1')
    if a:
        ri = a.get('rcsb_struct_symmetry') or []
        out['oligomeric_state'] = (a.get('pdbx_struct_assembly') or {}).get('oligomeric_details', '?')
        out['oligomeric_count'] = (a.get('pdbx_struct_assembly') or {}).get('oligomeric_count', '?')
        out['symmetry'] = ri[0].get('symbol', '?') if ri else '?'
    # ligands
    ligs, metals = [], []
    for i in range(1, int(out.get('nonpolymer_entities') or 0) + 1):
        n = get(f'https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb}/{i}')
        if not n: continue
        cid = (n.get('pdbx_entity_nonpoly') or {}).get('comp_id', '?')
        if cid in ADDITIVES: continue
        (metals if cid in METALS else ligs).append(cid)
    out['ligands'] = ligs
    out['metals'] = metals
    return out

def main():
    prots = sys.argv[1:] if len(sys.argv) > 1 else [
        '2K5H','6EWT','1W4H','6EWS','1GYZ','1QP2','1TUC','1PSE','1QKH','2KWH',
        '2K1B','2KXD','2KVS','2LQK','2PTL','2RU9','6OBK','2L9R','6M3N','2KYB','1I6C']
    print(f"{'pdb':<7}{'method':<10}{'chains':>7}{'oligomer':>22}{'ligands':>16}{'metals':>10}")
    print('-' * 74)
    rows = []
    for p in prots:
        r = annotate(p); rows.append(r)
        if r.get('status') != 'ok':
            print(f"{p:<7}{'NOT FOUND':<10}"); continue
        print(f"{p:<7}{str(r.get('method'))[:9]:<10}{r.get('chains',0):>7}"
              f"{str(r.get('oligomeric_state','?'))[:21]:>22}"
              f"{','.join(r['ligands'])[:15] or '-':>16}{','.join(r['metals'])[:9] or '-':>10}")
        time.sleep(0.15)
    ok = [r for r in rows if r.get('status') == 'ok']
    nl = sum(1 for r in ok if r['ligands']); nm = sum(1 for r in ok if r['metals'])
    multi = sum(1 for r in ok if (r.get('chains') or 0) > 1)
    print(f"\n=== SUMMARY over {len(ok)} resolved structures ===")
    print(f"  with a real ligand (additives excluded): {nl}")
    print(f"  with a bound metal:                      {nm}")
    print(f"  multi-chain in the deposited entry:      {multi}")
    print(f"  monomeric single-chain:                  {len(ok)-multi}")
    if nl == 0 and nm == 0:
        print("\n  VERDICT: zero ligands and zero metals across our test set.")
        print("  The ligand/metal levers are UNMEASURABLE here -- a constant-zero column")
        print("  contributes nothing to any gradient. This is a property of the BENCHMARK,")
        print("  not evidence against the idea.")
    else:
        print(f"\n  VERDICT: {nl} ligand-bearing and {nm} metal-bearing structures exist.")
        print("  The levers ARE testable on this set. Report which proteins and re-plan.")
    json.dump(rows, open('/home/nissimb/DeepPEF/results/08_data/pdb_annotations.json', 'w'), indent=1)
    print("\nwrote results/08_data/pdb_annotations.json")

if __name__ == '__main__':
    main()
