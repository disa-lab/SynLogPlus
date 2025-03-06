#!/bin/bash

type=$1
if [[ "${type}" == "full" ]]; then
  param="--use_full"
else
  type="2k"
  param=""
fi

# echo $type
# echo $param

# for technique in AEL Drain IPLoM LFA LogCluster Logram LogSig MoLFI SHISO # Spell
# for technique in AEL Drain IPLoM LenMa LFA LogCluster LogMine Logram LogSig MoLFI SHISO SLCT Spell \
# for technique in AEL Drain # GeLL # IPLoM LFA LogCluster Logram SHISO \
  # LogPPT UniParser
for technique in GeLL
do
    echo ${technique}
    python evaluator.py --dirpath ../result/result_${technique}_${type}/ \
      --pathformat "{}_${type}.log_structured.csv" ${param} # 2>&1>/dev/null &
done
wait
