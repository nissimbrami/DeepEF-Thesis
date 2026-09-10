import io, ast, shutil, re
HN="model/hydro_net.py"
shutil.copy(HN+".bak_b1d", HN)          # restore the pristine file
hn=io.open(HN,encoding="utf-8").read()

# 1. copy the head class in, before class PEM(
src=io.open("scripts_new/distogram_head.py",encoding="utf-8").read()
a=src.index("class DistogramHead"); b=src.index("# ====",a)
head=src[a:b].rstrip()+"\n"
hn=hn.replace("class PEM(", head+"\n\n"+"class PEM(",1)

# 2. instantiate inside PEM.__init__, immediately after its def line
i=hn.index("class PEM(")
m=re.search(r"\n(    def __init__\(self[^\n]*\n(?:[^\n]*\n)*?)(    def )", hn[i:])
assert m, "PEM.__init__ not located"
init_block=m.group(1)
# find the last line of __init__ that is indented 8 spaces
lines=init_block.split("\n")
last=0
for k,l in enumerate(lines):
    if l.startswith("        ") and l.strip(): last=k
ins=("        # B1d: distogram auxiliary head on the UNFOLDED state (IFUM's learned half).\n"
     "        # Built ONLY when the weight is > 0, so weight 0 is bit-identical to baseline.\n"
     "        self.distogram_weight = float(getattr(CFG, 'distogram_weight', 0.0))\n"
     "        self.distogram_head = (DistogramHead(d_model=128)\n"
     "                               if self.distogram_weight > 0 else None)")
lines.insert(last+1, ins)
new_init="\n".join(lines)
hn=hn[:i]+hn[i:].replace(init_block,new_init,1)
io.open(HN,"w",encoding="utf-8").write(hn)
ast.parse(io.open(HN,encoding="utf-8").read())
print("hydro_net.py OK  DistogramHead:%d  self.distogram_head:%d"
      % (hn.count("DistogramHead"), hn.count("self.distogram_head")))
ast.parse(io.open("Megascale-fineTuning/train.py",encoding="utf-8").read())
print("train.py OK  distogram_loss:%d"
      % io.open("Megascale-fineTuning/train.py",encoding="utf-8").read().count("distogram_loss"))
