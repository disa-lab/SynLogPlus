datasets = [
    "Android",
    "Apache",
    "BGL",
    "HDFS",
    "HPC",
    "Hadoop",
    "HealthApp",
    "Linux",
    "Mac",
    "OpenSSH",
    "OpenStack",
    "Proxifier",
    "Spark",
    "Thunderbird",
    "Windows",
    "Zookeeper",
]

for dataset in datasets:
    projects_dir = "/local/home/enan/projects"
    loghub_csv = f"{projects_dir}/loghub-2.0/2k_dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv"
    log3t_csv = f"{projects_dir}/Log3T/logs/{dataset}/{dataset}_2k.log_structured_corrected.csv"
    print(f"diff {loghub_csv} {log3t_csv}")
