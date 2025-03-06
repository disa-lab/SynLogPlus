# source ~/.zshrc
# conda activate logevaluate

# Logram LogSig MoLFI

cd evaluation
# for technique in AEL Drain IPLoM LenMa LFA LogCluster LogMine Logram LogSig MoLFI SHISO SLCT Spell
for technique in AEL Drain IPLoM LFA LogCluster Logram LogSig MoLFI SHISO Spell
# for technique in Logram LogSig MoLFI
for technique in Spell
do
    echo ${technique}
    python ${technique}_eval.py -full
done
