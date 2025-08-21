import sys
import os
from tqdm.auto import tqdm

sys.path.append('../')
from old_benchmark.Drain_benchmark import benchmark_settings
from evaluation.utils.common import common_args, unique_output_dir
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average

from DrainPlus import LogParser

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
    args = common_args()
    data_type = "full" if args.full_data else "2k"
    input_dir = f"../../{data_type}_dataset/"
    output_dir = f"../../result/result_DrainPlus_{data_type}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    datasets = datasets_full if args.full_data else datasets_2k

    # dataset = 'OpenSSH'
    for dataset in datasets:
        # if dataset not in ['Proxifier']: continue
        setting = benchmark_settings[dataset]
        log_file = setting['log_file'].replace("_2k", f"_{data_type}")
        # log_file = log_file.replace(data_type, f"{data_type}-{split}")
        indir = os.path.join(input_dir, os.path.dirname(log_file))
        parser = LogParser(setting['log_format'], indir, output_dir, setting['depth'], setting['st'], rex=setting['regex'])
        parser.parse(os.path.basename(log_file))
