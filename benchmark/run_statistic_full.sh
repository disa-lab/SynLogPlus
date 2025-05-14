#!/bin/bash

cd evaluation
# for technique in AEL Drain IPLoM LenMa LFA LogCluster LogMine Logram LogSig MoLFI SHISO SLCT Spell
for technique in AEL Drain IPLoM LFA LogCluster Logram LogSig MoLFI SHISO Spell LenMa
do
    echo $(date)
    echo ${technique}
    python ${technique}_eval.py -full
done
