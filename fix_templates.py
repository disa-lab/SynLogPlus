import pandas as pd
from pathlib import Path

rootdir = Path('.')

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

datatype = 'full'
logdir = rootdir / "{}_dataset".format(datatype)

datasets = datasets_full if datatype=='full' else datasets_2k
suffix = "" if datatype=='full' else '_corrected'

for dataset in datasets:
    # if dataset!='Windows': continue

    print(dataset)
    csvpath = logdir / dataset / f"{dataset}_{datatype}.log_structured{suffix}.csv"
    df = pd.read_csv(csvpath)
    df.to_csv(csvpath.with_suffix('.csv.bak'), index=False)

    # df['EventTemplate'] = df['EventTemplate'].str.replace("@<\*>", "<*>", regex=True)
    df['EventTemplate'] = df['EventTemplate'].str.replace("(<\*>[@:/-])+<\*>", "<*>", regex=True)
    df['EventTemplate'] = df['EventTemplate'].str.replace("(<\*>,? ?)+<\*>", "<*>", regex=True)
    df['EventTemplate'] = df['EventTemplate'].str.replace("(<\*> ?)+<\*>", "<*>", regex=True)
    df['EventTemplate'] = df['EventTemplate'].str.replace("(<\*> ?)([KMGkmg][Bb](ytes)?)", "<*>", regex=True)
    df.to_csv(csvpath, index=False)

#     templates = df['EventTemplate'].unique()
#     for template in templates:
#         print(template)
#     print()
