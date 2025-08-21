import sys
import os
from tqdm.auto import tqdm

sys.path.append('../')
from old_benchmark.Drain_benchmark import benchmark_settings
from evaluation.utils.common import common_args, unique_output_dir
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average

from logparser.Drain.Drain3 import LogParser
import drain3
from drain3.template_miner_config import TemplateMinerConfig as Drain3Config

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
    output_dir = f"../../result/result_Drain3_{data_type}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = prepare_results(
        output_dir=output_dir,
        otc=args.oracle_template_correction,
        complex=args.complex,
        frequent=args.frequent
    )

    if args.full_data:
        datasets = datasets_full
    else:
        datasets = datasets_2k

    # dataset = 'OpenSSH'
    for dataset in datasets:
        setting = benchmark_settings[dataset]
        log_file = setting['log_file'].replace("_2k", f"_{data_type}")
        # log_file = log_file.replace(data_type, f"{data_type}-{split}")
        indir = os.path.join(input_dir, os.path.dirname(log_file))
        if os.path.exists(os.path.join(output_dir, f"{dataset}_{data_type}.log_structured.csv")):
            parser = None
            print("parseing result exist.")
        else:
            parser = LogParser

        parser = LogParser(setting['log_format'], indir, output_dir, setting['depth'], setting['st'], rex=setting['regex'])
        parser.parse(os.path.basename(log_file))
