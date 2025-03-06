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

logdir = rootdir / "2k_dataset"

for dataset in datasets_2k:
    if dataset!='Windows': continue

    print(dataset)
    csvpath = logdir / dataset / f"{dataset}_2k.log_structured_corrected.csv"
    df = pd.read_csv(logdir / dataset / f"{dataset}_2k.log_structured_corrected.csv")
    df.to_csv(csvpath.with_suffix('.csv.bak'), index=False)

    # df['EventTemplate'] = df['EventTemplate'].str.replace("@<\*>", "<*>", regex=True)
    df['EventTemplate'] = df['EventTemplate'].str.replace("(<\*>[@:/-])+<\*>", "<*>", regex=True)
    df['EventTemplate'] = df['EventTemplate'].str.replace("(<\*>,? ?)+<\*>", "<*>", regex=True)
    df['EventTemplate'] = df['EventTemplate'].str.replace("(<\*> ?)+<\*>", "<*>", regex=True)
    df.to_csv(csvpath, index=False)

#     templates = df['EventTemplate'].unique()
#     for template in templates:
#         print(template)
#     print()
