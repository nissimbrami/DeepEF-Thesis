import os, sys, glob, torch, datetime
sys.path.insert(0,'/home/nissimb/DeepPEF')
base='/home/nissimb/DeepPEF/Megascale-fineTuning/models'
allck=sorted(glob.glob(base+'/*/kf_*_epoch_*.pt'))
print("total ckpt files on disk:", len(allck))
legacy=[]; wrapper=[]; bad=[]
for p in allck:
    try:
        o=torch.load(p,map_location='cpu',weights_only=False)
    except Exception as e:
        bad.append((p,str(e)[:80])); continue
    if isinstance(o,dict) and 'model_state_dict' in o: wrapper.append((p,o))
    else: legacy.append(p)
print("legacy bare state_dict:", len(legacy))
print("new wrapper format   :", len(wrapper))
print("unreadable           :", len(bad))
for p,e in bad[:5]: print("   UNREADABLE",p,e)
if wrapper:
    print("\nWRAPPER FILES (written by PATCHED code):")
    for p,o in wrapper[:12]:
        print("  ",p)
        print("      keys:",sorted(o.keys()))
        print("      mtime:",datetime.datetime.fromtimestamp(os.path.getmtime(p)))
if legacy:
    print("\nsample legacy:",legacy[0])
    print("      mtime:",datetime.datetime.fromtimestamp(os.path.getmtime(legacy[0])))
