#!/bin/bash

technique="IPLoM"
type="full"
param="-full"

# python evaluator.py --dirpath ../result/result_${technique}_${type}/ \
#   --pathformat "{}_${type}.log_structured.csv" ${param} # 2>&1>/dev/null &


cd evaluation
# for technique in AEL Drain IPLoM LenMa LFA LogCluster LogMine Logram LogSig MoLFI SHISO # SLCT Spell
for technique in Drain
do
    echo ${technique}
    python GeLL_eval.py -g ${technique} ${param}
done
