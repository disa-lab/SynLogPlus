#!/bin/bash

type=$1
if [[ "${type}" == "full" ]]; then
  param="--use_full"
else
  type="2k"
  param=""
fi

echo $type
echo $param

techniques="GeLL-AEL GeLL-Drain GeLL-IPLoM GeLL-LFA GeLL-LogCluster GeLL-Logram GeLL-LogSig GeLL-MoLFI GeLL-SHISO GeLL-Spell"

for technique in ${techniques}
do
    echo ${technique}
    python evaluator.py --dirpath ../result/result_${technique}_${type}/ \
      --pathformat "{}_${type}.log_structured.csv" ${param} # --data Linux 2>&1>/dev/null &
done
wait
