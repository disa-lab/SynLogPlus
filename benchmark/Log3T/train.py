from Log3T import Log3T,preprocess
import sys
import torch
from pathlib import Path
from Transfomer_encoder import transfomer_encoder
from settings import benchmark_settings

# sys.path.append('../../')




import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-full", "--use_full", action="store_true")
args = parser.parse_args()
type = 'full' if args.use_full else '2k'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logdir = Path('../../{}_dataset'.format(type))
Path('torch_model').mkdir(parents=True, exist_ok=True)
for dataset, setting in benchmark_settings.items():
    if args.use_full and dataset in [ 'Android','Windows' ]: #, 'BGL','HDFS','Spark','Thunderbird' ]:
        continue
    # if dataset != 'Linux': continue

    model=transfomer_encoder.BERT()
    model.to(device)
    parse = preprocess.format_log(
        log_format=setting['log_format'],
        indir = str(logdir / dataset),
    )
    form = parse.format('{}_{}.log'.format(dataset,type))
    content = form['Content']
    arr = content.to_numpy()
    sentences = arr.tolist()
    history_cutoff = len(sentences) # 2000
    variablelist=Log3T.read_csv_to_list('Variableset/variablelist1'+dataset+'.csv')
    batch,log_sentence=Log3T.log_to_model(sentences[:history_cutoff],stage='train',regx=setting['regex'],regx_use=False,dataset=dataset,variablelist=variablelist)
    Log3T.train(batch,epoch_n=setting['epoch'],output=dataset,model=model)
    torch.save(model.state_dict(), f'torch_model/model_{dataset}_{type}')
