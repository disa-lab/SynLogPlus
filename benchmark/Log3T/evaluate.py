import re
import os
from pathlib import Path
import datetime
from Log3T import Log3T
from Log3T import preprocess
import pandas as pd
import sys
import torch
from Transfomer_encoder import transfomer_encoder
from settings import benchmark_settings

# sys.path.append('../../')


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-full", "--use-full", action="store_true")
parser.add_argument("--group-first", action="store_true")
parser.add_argument("--eval-grouping", action="store_false")
parser.add_argument("--eval-training", action="store_false")
args = parser.parse_args()

type = 'full' if args.use_full else '2k'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_training = args.eval_grouping
eval_grouping = args.eval_training
do_grouping_first = args.group_first

print(f"Evaluating only grouping: {eval_grouping}")
print(f"Evaluating only grouping: {eval_training}")
print(f"Do grouping first: {do_grouping_first}")

resultdir = 'Result-group-first' if do_grouping_first else 'Result-'
if eval_grouping and eval_training:
    resultdir += 'two-phase/'
elif eval_grouping:
    resultdir +=  'only-grouping/'
elif eval_training:
    resultdir +=  'only-training/'

Path(resultdir).mkdir(parents=True, exist_ok=True)

logdir = Path('../../{}_dataset'.format(type))
for dataset, setting in benchmark_settings.items():
    if args.use_full and dataset in [ 'Android','Windows', ]: # 'BGL','HDFS','Spark','Thunderbird' ]:
        continue
    # if dataset!='Proxifier': continue
    model=transfomer_encoder.BERT()
    model1=transfomer_encoder.BERT()
    if eval_training: model.load_state_dict(torch.load(f'torch_model/model_{dataset}_{type}'))
    start=datetime.datetime.now()
    parse = preprocess.format_log(
        log_format=setting['log_format'],
        indir = str(logdir / dataset),
    )
    form = parse.format(f"{dataset}_{type}.log")
    content = form['Content']
    template=pd.DataFrame()
    sentences = content.tolist()
    print(dataset)
    log_data,log_sentence=Log3T.log_to_model(sentences,stage='parse',regx=[],regx_use=False,dataset=dataset,variablelist=[]) # 如果你想使用过滤器，可以在regx=""处添加,并且将regx_use改为True,过滤器的设计可以根据错误解析的结果进行设定
    log_group,group_with_template,predict_label,partial_constants=Log3T.parse(log_data,log_sentence, setting['threshold'],{},0,model,model1,do_grouping_first)

    thres = 0.9

    templates = sentences.copy()
    if eval_grouping:
        for key in group_with_template.keys():
            list_mem=group_with_template[key]
            for id in list_mem:
                templates[id]=key
    elif eval_training:

        for idx in range(len(content)):
            templates[idx] = " ".join([
                x
                if predict_label[idx][i] < thres and not Log3T.is_number(x)
                else "<*>"
                for i, x in enumerate(preprocess.wordsplit(sentences[idx],dataset))
            ])

    end = datetime.datetime.now()
    print('time taken == #'+str(end-start))

    form['EventTemplate']=templates
    form.to_csv(f'{resultdir}/{dataset}_{type}.log_structured.csv')
