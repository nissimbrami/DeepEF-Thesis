"""Build data/het_classification.csv from the catalogue's Ligands_y column.

Classification follows PDB chemical component conventions (wwPDB CCD component `type`
plus the standard crystallization-additive lists). Categories:

  crystallization_additive : buffer / precipitant / cryoprotectant. Present because of how
                             the crystal was grown, NOT because the protein binds it in vivo.
  metal                    : free metal ion / metal cluster.
  cofactor                 : tightly bound non-protein functional group (HEM, FAD, NAD...).
  substrate                : substrate / product / analogue / inhibitor genuinely bound.
  modified_residue         : an amino acid IN the polypeptide chain (MSE, SEP, TPO...).
  polymer_residue          : a nucleotide IN a DNA/RNA chain (A, U, DG, DT...).
  unknown                  : unclassified, or a deliberately-unknown CCD code (UNK/UNX/UNL).

modified_residue and polymer_residue are NOT ligands: they are monomers of a covalent
polymer. Counting them as bound ligands is a category error (MSE = selenomethionine,
an amino acid used for phasing; DA = a deoxyadenosine residue of a DNA duplex).
"""
import csv, os, re, sys, collections

# ---------------------------------------------------------------- category tables
METAL = """MG ZN CA NA MN K FE CU CD NI CO HG BR IOD LI SR PT FE2 CU1 AU AG BA CS RB
PB TL SM EU GD YB HO3 LU W MO V CR SE4 3NI 2HP ZN2 CUA OEX OS IR RU PD PR CE LA
NAO NAW MOO YT3 AL GA IN TB Y1 ER3 NA6 K1 FE3 MN3 CO3H""".split()
CLUSTER = "SF4 FES F3S FS4 CLF CFN NFS ICS CUB HC0 HC1 XCC WCC".split()

ADDITIVE = """SO4 GOL CL EDO PO4 ACT PEG MPD TRS PG4 FMT PGE MES ACY EPE CIT BME
1PE IMD IPA NO3 DMS FLC MRD P6G CAC MLI TLA SCN BCT AZI DIO NHE EOH BTB DTT SIN
BEN NH4 HED HEZ PGO SO3 2PE OXL TAM TAR MLT B3P CXS BU3 PG0 LDA C8E BOG LMT DMU
LMU HTG UMQ CPS TAU PEO P33 12P 15P 7PE DOD OH OHX PE4 SPD PUT SRM CHD BEZ PGR
XPE MYR PLM OLA OLC CLR MPO POL ACN TFA CAP SPM SPK EDT NHS ETX ACE NH2 DOX
PDO IPH TBU MOH DMF NH3 PIN BTB TMA GBL PYR ACD BAM DEP AKG3 GLYCEROL""".split()

COFACTOR = """HEM FAD NAD NAP NDP FMN SAH SAM COA HEC PLP GSH B12 TPP MTE HEA
BCL CLA BCR PQN U10 RET HEV DPM SF4C ACO FDA NAI COO TDP MGD H4B BIO F43 COB
CO1 HDD HAS SRM1 PHO PEB CYC LLP MDO F42 THF FOL MQ7 MQ8 UQ1 CIT1 GTX ADN0
LMG DGD SQD CDL PEE PGV LHG PL9 CHL PHY""".split()

SUBSTRATE = """NAG BMA MAN ADP GDP FUC GLC GAL ATP ANP BGC AMP GTP GNP NDG SIA
UDP NGA A2G FUL FRU XYP XYS KDO GLA ADN UMP CMP 5GP GCP ACP AGS POP CMO OXY
STU PAR THP UPG DUP GSP P5P Y5P U5P DOC 8OG GUN ADE ABA NLE PYR SIN1 MLA
BEF ALF AF3 CYN NO NGK MAL SUC TRE LAT LBT CBI ARB RIB RAM SGN IDS GCU BDP
G6P F6P FBP 2PG 3PG PEP OAA MAL1 CIT2 ICT SUCC FUM AKG NADH PYR1 LAC
0TD 0QE ZBA SMA CRO CYS1 PAP APR IMP XMP TMP DTP DUT CTP UTP""".split()

MODRES = """MSE SEP TPO PTR CSO CSD KCX PCA CME MLY M3L ALY CGU HIC OCS CSX CAS
CSS SMC MLZ TYS HYP FME DAL MEN ABA1 SAC AYA CXM SCY CSW OMT NEP HIP CIR
DDE 2MR AGM SNN SNC OAS MHO SME CY3 CSU CAF NMM LYZ TRO TYI TYQ ORN DPR
BMT ALO AIB SAR MVA DAR DGL DSN DTH DTY DVA DPN DLE DLY DIL DAS DCY DHI
0TD1 CSP CGN PHD PN2 TPQ CRO1 NRQ GYS CH6 CQR CR2 CR8 DYG DYA""".split()

# nucleotide residues of DNA/RNA chains (CCD polymer components)
POLYRES = """A C G U DA DC DG DT DU DI I PSU 5MU OMG 5MC 2MG UR3 M2G 7MG 1MA H2U
OMC OMU MA6 2MA 4OC 1MG 6MZ G7M 4SU 5BU 5IU CBR CFL 3DR N 12A A23 AET G46
2MU 3MU 70U MIA T6A YG YYG QUO 5FU 5CM 6MA 1RN P5P1 IU SUR MTU 3TD 2AU
UFT AS DDG DDN 8MG G48 A2M CCC 5AA U8U OMI 2SU""".split()

UNKNOWN_EXPLICIT = "UNK UNX UNL ERROR NONE NAN UNL1 DUM".split()

CAT = {}
def put(codes, cat):
    for c in codes:
        c = c.strip().upper()
        if c and c not in CAT:
            CAT[c] = cat

# order matters: most specific / least ambiguous first
put(UNKNOWN_EXPLICIT, "unknown")
put(MODRES, "modified_residue")
put(POLYRES, "polymer_residue")
put(METAL, "metal"); put(CLUSTER, "metal")
put(COFACTOR, "cofactor")
put(ADDITIVE, "crystallization_additive")
put(SUBSTRATE, "substrate")

NOTES = {
 "SO4":"sulfate - crystallization precipitant","GOL":"glycerol - cryoprotectant",
 "EDO":"ethylene glycol - cryoprotectant","MSE":"selenomethionine - amino acid IN the chain (MAD/SAD phasing), not a ligand",
 "SEP":"phosphoserine - modified residue in the chain","TPO":"phosphothreonine - modified residue in the chain",
 "PTR":"phosphotyrosine - modified residue in the chain",
 "A":"adenosine residue of an RNA chain, not a bound ligand","U":"uridine residue of an RNA chain",
 "C":"cytidine residue of an RNA chain","G":"guanosine residue of an RNA chain",
 "DA":"deoxyadenosine residue of a DNA chain","DT":"thymidine residue of a DNA chain",
 "DC":"deoxycytidine residue of a DNA chain","DG":"deoxyguanosine residue of a DNA chain",
 "HEM":"heme b - cofactor","FAD":"flavin adenine dinucleotide - cofactor",
 "NAD":"NAD+ - cofactor","UNK":"deliberately unknown atom/residue in the CCD",
 "UNX":"unknown atom or ion","UNL":"unknown ligand","PEG":"polyethylene glycol - precipitant",
 "MPD":"2-methyl-2,4-pentanediol - precipitant/cryo","ACT":"acetate ion - buffer",
 "PO4":"phosphate ion - buffer (also a genuine substrate in some enzymes; ambiguous, scored as additive)",
 "CL":"chloride ion - buffer salt","NAG":"N-acetylglucosamine - glycosylation/substrate",
 "SF4":"iron-sulfur [4Fe-4S] cluster","FES":"[2Fe-2S] cluster",
}

# ------------------------------------------------------------------- build
freq_path = sys.argv[1] if len(sys.argv) > 1 else "het_freq.tsv"
out_path  = sys.argv[2] if len(sys.argv) > 2 else "data/het_classification.csv"

pairs = []
with open(freq_path) as fh:
    for line in fh:
        line = line.strip()
        if not line: continue
        code, n = line.rsplit("\t", 1)
        pairs.append((code.upper(), int(n)))
pairs.sort(key=lambda kv: -kv[1])
total_occ = sum(n for _, n in pairs)

rows, covered = [], 0
for rank, (code, n) in enumerate(pairs, 1):
    cat = CAT.get(code, "unknown")
    assigned = code in CAT and cat != "unknown"
    if assigned: covered += n
    bound = "yes" if cat in ("metal", "cofactor", "substrate") else "no"
    rows.append({
        "het_code": code, "rank": rank, "occurrences": n,
        "classification": cat,
        "counts_as_bound_ligand": bound,
        "curated": "yes" if code in CAT else "no",
        "note": NOTES.get(code, ""),
    })

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["het_code","rank","occurrences","classification",
                                       "counts_as_bound_ligand","curated","note"])
    w.writeheader(); w.writerows(rows)

top200 = pairs[:200]
top200_cov = sum(n for c, n in top200 if c in CAT) / sum(n for _, n in top200)
print(f"wrote {out_path}: {len(rows)} codes, {total_occ} occurrences")
print(f"curated codes present in table : {sum(1 for r in rows if r['curated']=='yes')}")
print(f"OCCURRENCE coverage (curated)  : {covered}/{total_occ} = {covered/total_occ:.4f}")
print(f"top-200 occurrence coverage    : {top200_cov:.4f}")
agg = collections.Counter()
for r in rows: agg[r["classification"]] += r["occurrences"]
for k, v in agg.most_common():
    print(f"  {k:26s} {v:7d}  {100*v/total_occ:5.2f}%")
nonbound = sum(r["occurrences"] for r in rows if r["counts_as_bound_ligand"]=="no")
print(f"NOT a bound ligand (additive+modres+polyres+unknown): {100*nonbound/total_occ:.2f}%")
