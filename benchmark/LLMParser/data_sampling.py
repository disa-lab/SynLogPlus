import re
import random
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.cluster import MeanShift
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--systems",
    type=str,
    default="Proxifier,Linux,Apache,Zookeeper,Mac,Hadoop,OpenStack,HealthApp,HPC,OpenSSH,BGL,HDFS,Spark,Thunderbird,Android,Windows",
)
parser.add_argument("--shot", type=int, default=50)
parser.add_argument("--full", action='store_true')
args = parser.parse_args()

random.seed(41)


dtype = 'full' if args.full else '2k'
suffix = '' if args.full else '_corrected'

logdir = '../../full_dataset' if dtype== 'full' else 'logs'

def replace_numbers_with_zero(text):
    return re.sub(r'\d+(\.\d+)?', '0', text)

def train_data_sample(project, shot):
    dataset_path = f"{logdir}/{project}/{project}_{dtype}.log_structured{suffix}.csv"
    # load the dataset and make statistics
    keep_columns = ["LineId", "Content", "EventTemplate"]
    raw_dataset = pd.read_csv(dataset_path, index_col=False, usecols=keep_columns)
    # print(raw_dataset)
    raw_dataset = raw_dataset.map(str)

    _n = '2000h'
    raw_dataset = raw_dataset.head(int(_n[:-1]))
    # _n = 2000
    # if len(raw_dataset) > _n:
    #     raw_dataset = raw_dataset.sample(n=_n)

    # Extract the text column
    raw_dataset['Content_0'] = raw_dataset['Content'].apply(replace_numbers_with_zero)
    text_column = raw_dataset['Content_0']

    # Text preprocessing and vectorization
    vectorizer = TfidfVectorizer()
    data_matrix = vectorizer.fit_transform(text_column).toarray()

    # Mean Shift clustering
    mean_shift = MeanShift(bandwidth=0.5)
    clusters = mean_shift.fit_predict(data_matrix).tolist()
    content_list = raw_dataset['Content'].tolist()
    cluster_dict = {}
    for data, cluster_id in zip(content_list, clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(data)
    sorted_clusters = sorted(cluster_dict.values(), key=len, reverse=True)
    sampled_log = []
    while len(sampled_log) < shot:
        for i in sorted_clusters:
            if len(sampled_log) == shot:
                break
            if i != []:
                sample = random.choice(i)
                sampled_log.append(sample)
                i.remove(sample)
    # label result
    template_list = []
    for element in sampled_log:
        value = raw_dataset.loc[raw_dataset["Content"] == element, 'EventTemplate'].values[0]
        template_list.append(value)
    data = {'input': sampled_log,
            'output': template_list}
    res_df = pd.DataFrame(data)
    res_df.insert(0, 'instruction', "Parse the input log to log template.")
    save_path = f"./training_samples_{dtype}/{str(shot)}_{_n}/{project}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    res_df.to_json(save_path + "train.json", orient="records")


project_list = args.systems.split(",")

if dtype == 'full':
    exclusion_list = ['Android','Windows','Spark','BGL','HDFS','Thunderbird']
    project_list = [ p for p in project_list if p not in exclusion_list ]

for project in project_list:
    train_data_sample(project=project, shot=args.shot)
