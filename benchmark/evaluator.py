import multiprocessing as mp
import numpy as np
import pandas as pd
import scipy.special
from nltk.metrics.distance import edit_distance as calc_edit_distance

def get_accuracy(series_groundtruth, series_parsedlog, dataset, queue):
    correctly_parsed_logs = series_groundtruth.eq(series_parsedlog).values.sum()
    total_logs = len(series_groundtruth)
    PA = float(correctly_parsed_logs) / total_logs

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
    parser.add_argument("--dirpath", type=str, default=None)
    parser.add_argument("--use_full", action="store_true")
    parser.add_argument("--pathformat", type=str, default="{}/predictions.csv")
    parser.add_argument(
        "--datasets",
        type=str,
        default="Android,Apache,BGL,HDFS,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Spark,Thunderbird,Windows,Zookeeper"
        # default="Android,Apache,HPC,Hadoop,HealthApp,Linux,Mac,OpenSSH,OpenStack,Proxifier,Windows,Zookeeper"
    )
    args = parser.parse_args()
    data_type = 'full' if args.use_full else '2k'

    dirpath = Path(args.dirpath)

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
    for dataset in datasets:
        if args.use_full and dataset in [
                'Android', 'Windows', 'BGL', 'HDFS', 'Spark', 'Thunderbird'
        ]: continue
        print(dataset)

        dataset_dir = Path(__file__).absolute().parent.parent / '{}_dataset'.format(data_type)
        predic_file = dirpath / args.pathformat.format(dataset)
        ground_file = dataset_dir / dataset / '{}_2k.log_structured_corrected.csv'.format(dataset)
        ground_file = ground_file if not args.use_full else ground_file.parent / '{}_full.log_structured.csv'.format(dataset)

        # print(dataset_dir)
        # print(predic_file)
        # print(ground_file)
        # exit()

        if not predic_file.is_file():
            print(f"Predictions for   {dataset}   not found: skipping")
            continue

        df_parsedlog = pd.read_csv(predic_file, index_col=False, header='infer', dtype=str).map(str)
        df_groundtruth = pd.read_csv(ground_file)
        series_parsedlog = df_parsedlog['EventTemplate'] #.str.replace( r"\s+", "", regex=True)
        series_groundtruth = df_groundtruth['EventTemplate'] #.str.replace( r"\s+", "", regex=True)
        assert(len(series_groundtruth) == len(series_parsedlog)), f"{len(series_groundtruth)}, {len(series_parsedlog)}"

        t = mp.Process(target=get_accuracy, args=(series_groundtruth, series_parsedlog, dataset, q))
        tasks.append(t)
        t.start()

    for task in tasks:
        task.join()

    while not q.empty():
        accuracies = accuracies | q.get()

    # print(accuracies)

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
    df.loc['Average'] = df.mean()
    df[df.columns] = df[df.columns].map(lambda x: '{0:.03}'.format(x))
    df.to_csv(Path(dirpath) / 'results.csv')
    print(df)
