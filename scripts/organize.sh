#!/bin/bash
# Organise results/ into a navigable structure. Uses `git mv` so history follows the files,
# and never deletes: anything unclassified stays where it is.
cd /home/nissimb/DeepPEF/results
mkdir -p 01_start_here 02_findings 03_levers 04_data_integrity 05_infrastructure 06_planning 07_agent_reports figures

mv_ () { [ -f "$1" ] && git mv -f "$1" "$2/" 2>/dev/null || { [ -f "$1" ] && mv -f "$1" "$2/"; }; }

# 01 - read these first, in this order
for f in MASTER.md FINDINGS.md STATUS_FULL.md TASKS.md CONTEXT.md; do mv_ "$f" 01_start_here; done

# 02 - scientific findings, one per question
for f in THESIS_RESULTS.md SLOPE_ARM.md SLOPE_OBJECTIVE.md SLOPE_ORIGIN.md DG_ARM.md \
         W5_ON_DG.md INFO_LEVERS.md LORO_RESULT.md SEVERING_RESULT.md PER_MUTATION.md \
         SIDECHAIN_DESIGN.md TWO_PROTEINS.md IDENTIFIABILITY.md OFFSET_CORRECTOR.md \
         LENGTH_CONFOUND.md ENSEMBLE.md FACTORIAL_ANALYSIS.md CATALOGUE_VS_BP.md \
         MISSING_PROTEINS.md; do mv_ "$f" 02_findings; done

# 03 - lever design and specs
for f in DESIGN_EMBEDDINGS.md DESIGN_CHEMISTRY.md DESIGN_BIOLOGY.md DESIGN_COMBINATION.md \
         DESIGN_OFFSET_ARCH.md DESIGN_OPEN_ALPHABET.md LIGAND_SPEC.md W9_RETIRED.md \
         WHAT_WE_SEE.md SLOPE_ARM_READY.md SEVERING_READY.md LORO_READY.md \
         ANNOTATION_SPEC.md PLDDT_SPEC.md; do mv_ "$f" 03_levers; done

# 04 - data integrity
for f in INTEGRITY_AUDIT.md CATALOGUE_INTEGRITY.md CATALOGUE_USES.md HADAR_REPORT.md \
         MEGASCALE_RAW.md OFIR_THESIS_NOTES.md OFIR_TRANSFER.md; do mv_ "$f" 04_data_integrity; done

# 05 - infrastructure
for f in GOLDEN_QOS.md VERIFY_9_ITEMS.md CHECKPOINT_INTEGRITY.md COMPUTE_OPTIMISATION.md \
         METHODS_SURVEY.md LITERATURE.md; do mv_ "$f" 05_infrastructure; done

# 06 - historical planning
for f in RESULTS.md phase0_actionD_calib_diag.md catalogue_report.md; do mv_ "$f" 06_planning; done

# 07 - raw agent output
for f in AGENT_REPORT.md; do mv_ "$f" 07_agent_reports; done

# the ctx*.md checkpoint fragments are already concatenated into CONTEXT.md; archive them
mkdir -p 06_planning/checkpoint_fragments
for f in ctx*.md; do [ -f "$f" ] && mv_ "$f" 06_planning/checkpoint_fragments; done

echo "--- structure ---"
for d in 0*/; do printf "%-24s %s files\n" "$d" "$(ls $d 2>/dev/null|wc -l)"; done
echo "unfiled in results/: $(ls *.md 2>/dev/null|wc -l) md, $(ls *.json 2>/dev/null|wc -l) json"
