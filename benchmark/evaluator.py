import csv
import multiprocessing as mp
import numpy as np
import pandas as pd
import scipy.special
from nltk.metrics.distance import edit_distance as calc_edit_distance

def get_accuracy(series_groundtruth, series_parsedlog, dataset, queue):
    correctly_parsed_logs = series_groundtruth.eq(series_parsedlog).values.sum()
    total_logs = len(series_groundtruth)
    PA = float(correctly_parsed_logs) / total_logs
    # print(PA, total_logs, correctly_parsed_logs, total_logs - correctly_parsed_logs)
    # exit()

    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])

    correctly_grouped_events = 0
    correctly_grouped_templates = 0
    correctly_parsed_templates = 0

    for groundtruth_template, group in df_combined.groupby('groundtruth'):
        parsed_templates_of_group = list(group['parsedlog'].unique())
        # print(parsed_templates_of_group)
        if len(parsed_templates_of_group) == 1:
            if len(group) == series_parsedlog[series_parsedlog == parsed_templates_of_group[0]].size:
                correctly_grouped_events += len(group)
                correctly_grouped_templates += 1
                if parsed_templates_of_group[0] == groundtruth_template:
                    correctly_parsed_templates += 1

    GA = correctly_grouped_events / len(series_groundtruth)
    PGA = float(correctly_grouped_templates) / len(series_parsedlog_valuecounts)
    RGA = float(correctly_grouped_templates) / len(series_groundtruth_valuecounts)
    PTA = float(correctly_parsed_templates) / len(series_parsedlog_valuecounts)
    RTA = float(correctly_parsed_templates) / len(series_groundtruth_valuecounts)

    FGA = 0.0 if PGA == 0 and RGA == 0 else 2 * (PGA * RGA) / (PGA + RGA)
    FTA = 0.0 if PTA == 0 and RTA == 0 else 2 * (PTA * RTA) / (PTA + RTA)

    queue.put({
        dataset: {
            'GA': GA,
            'PA': PA,
            'FGA': FGA,
            'FTA': FTA,
        }
    })

if __name__ == "__main__":

    import argparse
    from pathlib import Path
    rootdir = Path(__file__).absolute().parent.parent
    print(f"Project root: {rootdir}")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dirpath", type=str, default=None)
    parser.add_argument("-full", "--use_full", action="store_true")
    parser.add_argument("-p", "--pathformat", type=str, default="{}_{}.log_structured.csv")
    parser.add_argument(
        "--datasets",
        type=str,
        default="Android,Apache,BGL,HDFS,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Spark,Thunderbird,Windows,Zookeeper"
        # default="Android,Apache,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Windows,Zookeeper"
    )
    parser.add_argument(
        "-x", "--exclude_training_samples", action="store_true"
    )
    parser.add_argument(
        "-ts", "--training_samples_file", type=str, default=None
    )
    parser.add_argument(
        "-is", "--ignore_space_mismatch", action="store_true",
    )
    args = parser.parse_args()

    dirpath = Path(args.dirpath)
    data_type = 'full' if args.use_full else '2k'

    GA_list = []
    PA_list = []
    ED_list = []
    FGA_list = []
    FTA_list = []

    datasets = args.datasets.split(",")
    parsed_projects = []
    tla = {}
    mla = {}
    accuracies = {}
    tasks = []
    q = mp.Queue()
    flag = False
    for dataset in datasets:
        # if dataset != 'Apache': continue
        if args.use_full and dataset in [ 'Android','Windows', 'Spark', 'HDFS', 'BGL', 'Thunderbird' ]: continue
        if flag: print(dataset)

        dataset_dir = Path('/local/home/enan/projects/loghub-2.0')
        dataset_dir /= "{}_dataset".format(data_type)
        predic_file = dirpath / args.pathformat.format(dataset, data_type)
        result_file = dirpath / args.pathformat.format(dataset,data_type).replace(".csv","_result.csv")
        ground_file = dataset_dir / dataset / '{}_{}.log_structured{}.csv'.format(dataset, data_type, "" if args.use_full else "_corrected")

        # print(dataset_dir)
        # print(predic_file)
        # print(result_file)
        # print(ground_file)
        # exit()

        if not predic_file.is_file():
            if flag: print(f"Predictions for   {dataset}   not found: skipping")
            continue

        df_parsedlog = pd.read_csv(predic_file, index_col=False, header='infer', dtype=str)
        df_groundtruth = pd.read_csv(ground_file)
        if args.ignore_space_mismatch:
            series_parsedlog = df_parsedlog['EventTemplate'].str.replace( r"\s+", "", regex=True)
            series_groundtruth = df_groundtruth['EventTemplate'].str.replace( r"\s+", "", regex=True)
        else:
            series_parsedlog = df_parsedlog['EventTemplate'] #.str.replace( r"\s+", "", regex=True)
            series_groundtruth = df_groundtruth['EventTemplate'] #.str.replace( r"\s+", "", regex=True)
        series_content = df_groundtruth['Content']

        if args.exclude_training_samples:
            if args.training_samples_file:
                # LogPPT/datasets/{}/32shot/training_samples.csv
                # UniParser/full_annotations/{}/training_samples.csv
                # LLMParser/training_data_full/{}/training_samples.csv
                with open(args.training_samples_file.format(dataset), newline='') as f:
                    reader = csv.reader(f)
                    # training_samples_indices = [ int(idx) for idx in list(reader)[0] ]
                    # training_samples = series_content[training_samples_indices].tolist()
                    training_samples_indices = []
                    training_samples = [ log[0] for log in list(reader)]
                    # print(training_samples)
                    # exit()
            else:
                training_samples_indices = list(range(100))
                training_samples = series_content[training_samples_indices].tolist()
            training_samples_indices = [ i for i,content in enumerate(series_content) if content in training_samples ]

            # print(set(list(range(2000))) - set(training_samples_indices))
            if flag: print(len(series_groundtruth), len(series_groundtruth) - len(training_samples_indices))
            # continue

            series_parsedlog.drop(  training_samples_indices, inplace=True)
            series_groundtruth.drop(training_samples_indices, inplace=True)

            # series_parsedlog = series_parsedlog.take(  training_samples_indices)
            # series_groundtruth = series_groundtruth.take(training_samples_indices)

        if flag: print(f"{len(series_groundtruth)}, {len(series_parsedlog)}")
        assert(len(series_groundtruth) == len(series_parsedlog)), f"{len(series_groundtruth)}, {len(series_parsedlog)}"

        t = mp.Process(target=get_accuracy, args=(series_groundtruth, series_parsedlog, dataset, q))
        tasks.append(t)
        t.start()

    for task in tasks:
        task.join()

    while not q.empty():
        accuracies = accuracies | q.get()

    for project,accuracies in accuracies.items():
        parsed_projects.append(project)
        GA_list.append(accuracies['GA'])
        PA_list.append(accuracies['PA'])
        FGA_list.append(accuracies['FGA'])
        FTA_list.append(accuracies['FTA'])

    df = pd.DataFrame({
        "Dataset": parsed_projects,
        "GA": GA_list,
        "PA": PA_list,
        "FGA": FGA_list,
        "FTA": FTA_list
    })
    df.set_index('Dataset',inplace=True)
    df.sort_index(inplace=True)
    df.loc['Average'] = df.mean()
    df[df.columns] = df[df.columns].map(lambda x: '{0:.03}'.format(x))
    result_file = 'results-unseen.csv' if args.exclude_training_samples else 'results.csv'
    df.to_csv(Path(dirpath) / result_file)
    print(df)
