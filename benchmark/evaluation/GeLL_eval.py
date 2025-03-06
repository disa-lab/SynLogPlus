import os
import sys
import torch

sys.path.append('../')

from transformers import set_seed
from GeLL import *
from old_benchmark.Drain_benchmark import benchmark_settings
from evaluation.utils.common import common_args, unique_output_dir
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average

datasets_2k = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
    "Android","Windows",
]

datasets_full = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    # "Spark",
    # "Thunderbird",
    # "BGL",
    # "HDFS",
]


if __name__ == "__main__":
    set_seed(62)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = common_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--grouper', help="Set the grouper algorithm", default=None)
    args_,_ = parser.parse_known_args()
    grouper = args_.grouper
    data_type = "full" if args.full_data else "2k"
    input_dir = f"../../{data_type}_dataset/"
    output_dir = f"../../result/result_GeLL-{grouper}_{data_type}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.full_data:
        datasets = datasets_full
    else:
        datasets = datasets_2k
    # datasets = ['Linux']

    for dataset in datasets:
        print(dataset)
        setting = benchmark_settings[dataset]

        log_file = setting['log_file'].replace("_2k", f"_{data_type}")
        indir = os.path.join(input_dir, os.path.dirname(log_file))
        log_messages, log_templates = read_logs(input_dir, dataset, data_type=='full')

        history_cutoff = 100
        maxlen         = 128 # max number of words in a log
        word_maxlen    = 16 # max sub tokens in a word

        model = BERT()
        tokenizer = trf.BertTokenizerFast.from_pretrained("bert-base-uncased")
        threshold      = 0.8

        parser = LogParser(dataset, model, tokenizer, threshold, maxlen, word_maxlen)
        log_groups = parser.group_logs(Path(f'../../result/result_{grouper}_{data_type}/{dataset}_{data_type}.log_structured.csv'))
        predictions = parser.fix_templates(log_groups,log_messages)

        os.makedirs(output_dir, exist_ok=True)
        _df = pd.DataFrame(data=list(zip(log_messages,predictions)))
        _df.to_csv("{}/{}_{}.log_structured.csv".format(output_dir, dataset,data_type),
                   header=['Logs','EventTemplate'], index=False, quoting=csv.QUOTE_ALL)
