import copy
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-full", "--use_full", action="store_true")
args = parser.parse_args()
type = 'full' if args.use_full else '2k'

rootdir = Path(__file__).absolute().parent.parent
techniques = [
    "AEL", "Drain", "IPLoM", "LenMa", "LFA", "LogCluster", "LogMine", "Logram", "LogSig",
    "MoLFI", "SHISO", "SLCT", "Spell",
    "LogPPT", "UniParser",
    "Log3T", "GeLL",
]

series_GA  = []
series_PA  = []
series_FGA = []
series_FTA = []

series_lists = {
    "GA":  [],
    "PA":  [],
    "FGA": [],
    "FTA": [],
}
metrics = list(series_lists.keys())
print(metrics)

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
    # "HDFS",
    # "Spark",
    # "Thunderbird",
    # "BGL",
]

datasets = datasets_full if args.use_full else datasets_2k

result_dir = rootdir / 'result'
parsed_techniques = 0
for technique in techniques:
    print(technique)
    result_csv_path = result_dir / 'result_{}_{}'.format(technique,type) / 'results.csv'
    if not result_csv_path.exists():
        print(f"{technique}: no evaluation results found")
        continue
    result_csv = pd.read_csv(result_csv_path)
    result_csv.set_index(result_csv.columns[0], inplace=True)
    if len(result_csv) < len(datasets) + 1:
        continue
    for metric in metrics:
        series = result_csv[metric].rename(technique)
        series_lists[metric].append(series)
    parsed_techniques += 1

df = {}
for metric in metrics:
    _df = pd.DataFrame(series_lists[metric], columns=datasets+['Average']).T
    df[metric] = _df

metric_groups = [ ['GA','PA'], ['FGA', 'FTA' ] ]
with pd.ExcelWriter(result_dir / 'results_{}.xlsx'.format(type)) as writer:
    _df = copy.deepcopy(df)
    for metric_group in metric_groups:
        for metric in metric_group:
            df[metric].columns = [ "{}_{}".format(c,metric) for c in df[metric].columns ]
        dfc = pd.concat([df[metric] for metric in metric_group], axis=1)
        dfc.to_excel(writer, sheet_name="_".join(metric_group))

    # for metric in metrics:
    #     df = pd.DataFrame(series_lists[metric], columns=datasets+['Average']).T
    #     df.to_excel(writer, sheet_name=metric)

    workbook = writer.book
    format1 = workbook.add_format()
    format1.set_bold()
    for sheetname, worksheet in writer.sheets.items():
        length = len(_df[metrics[0]])
        # print(length)
        # exit()
        for row in range(2,length+2):
            s = chr(ord('B'))
            e = chr(ord('B') + parsed_techniques-1)
            worksheet.conditional_format('{}{}:{}{}'.format(s,row,e,row), {
                'type':     'top',
                'value':    1,
                'format':   format1,
            })

            s = chr(ord(e) + 1)
            e = chr(ord('B') + parsed_techniques*2-1)
            worksheet.conditional_format('{}{}:{}{}'.format(s,row,e,row), {
                'type':     'top',
                'value':    1,
                'format':   format1,
            })
