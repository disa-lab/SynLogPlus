#!/bin/bash

cd evaluation
for technique in AEL Drain IPLoM LenMa LFA LogCluster LogMine Logram LogSig MoLFI SHISO SLCT Spell
do
    echo $(date)
    echo ${technique}
    python ${technique}_eval.py -otc
done
