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
import time

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
    # datasets = ['HPC']

    # dataset = 'OpenSSH'
    for dataset in datasets:
    # for split in [1,2,3,4,5,6]:
        # print(f"\nSplit: {split}")
        setting = benchmark_settings[dataset]
        log_file = setting['log_file'].replace("_2k", f"_{data_type}")
        # log_file = log_file.replace(data_type, f"{data_type}-{split}")

        log_group_file = Path(f'../../result/result_{grouper}_{data_type}/{dataset}_{data_type}.log_structured.csv')
        if not log_group_file.exists():
            print(f"Grouper did not group {dataset}")
            continue

        start_time = time.time()
        parser = LogParser(dataset)
        log_messages, log_templates = parser.read_logs(input_dir, dataset, data_type=='full')
        log_groups = parser.group_logs(log_group_file)
        # log_groups = parser.group_logs(Path(f'../../result/result_{grouper}_{data_type}/{dataset}_{data_type}-{split}.log_structured.csv'))
        predictions = parser.fix_templates(log_groups,log_messages)
        print(f"Parsing time: {time.time() - start_time}")

        os.makedirs(output_dir, exist_ok=True)
        _df = pd.DataFrame(data=list(zip(log_messages,predictions)))
        _df.to_csv("{}/{}_{}.log_structured.csv".format(output_dir, dataset,data_type),
                   header=['Logs','EventTemplate'], index=False, quoting=csv.QUOTE_ALL)
