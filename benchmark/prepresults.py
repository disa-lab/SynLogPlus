import copy
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-full", "--use_full", action="store_true")
args = parser.parse_args()
type = 'full' if args.use_full else '2k'

# frequency: LFA, LogCluster, Logram
# heuristics: AEL, Drain, Spell,   IPLoM, MoLFI
# similarity: LogMine,LenMa,SHISO,LogSig

rootdir = Path(__file__).absolute().parent.parent
techniques = [
    "LFA", "LogCluster", "Logram",
    # "LenMa", "LogMine",
    "SHISO", "LogMine", "LenMa", #"IPLoM", #"LogMine", "LogSig",
    "AEL", "Drain", "Spell",
    "UniParser", "LogPPT",
    "LLMParser", # "LLMParser_1000", "LLMParser_1000h",
    "GeLL-Drain",
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
parsed_techniques = []
for technique in techniques:
    print(technique)
    result_csv_path = result_dir / 'result_{}_{}'.format(technique,type) / 'results.csv'
    if not result_csv_path.exists():
        print(f"{technique}: no evaluation results found")
        continue
    result_csv = pd.read_csv(result_csv_path)
    result_csv.set_index(result_csv.columns[0], inplace=True)
    if len(result_csv) < len(datasets) + 1:
        print(f"{technique}: length mismatch: {len(result_csv)}, {len(datasets)}")
        # continue
    for metric in metrics:
        series = result_csv[metric].rename(technique)
        series_lists[metric].append(series)
    parsed_techniques.append(technique)

df = {}
n_rows = len(datasets) + 1
maxvals = { }
for metric in metrics:
    _df = pd.DataFrame(series_lists[metric], columns=datasets+['Average']).T
    df[metric] = _df
    maxvals[metric] = [ row.max() for _,row in _df.iterrows() ]

metric_groups = [ ['GA','PA'], ['FGA', 'FTA' ] ]
with pd.ExcelWriter(result_dir / 'results_{}.xlsx'.format(type)) as writer:
    _df = copy.deepcopy(df)
    for metric_group in metric_groups:
        for metric in metric_group:
            df[metric].columns = [ "{}_{}".format(c,metric) for c in df[metric].columns ]
        dfc = pd.concat([df[metric] for metric in metric_group], axis=1)
        dfc = dfc.reindex(sorted(dfc.columns), axis=1)
        l = [ x+f'_{m}' for x in parsed_techniques[-3:] for m in metric_group ]
        dfc = dfc[ [c for c in dfc if c not in l ] + l ]
        # dfc.drop(index=['Average'], inplace=True)
        # dfc.sort_index(inplace=True)
        # dfc.loc['Average'] = dfc.mean()
        dfc.to_excel(writer, sheet_name="_".join(metric_group))

    workbook = writer.book
    format1 = workbook.add_format()
    format1.set_bold()
    for sheetname, worksheet in writer.sheets.items():
        for row in range(n_rows):
            s = chr(ord('B'))
            e = chr(ord('B') + len(parsed_techniques)*2-1)
            # if not e.isalpha():
            #     e = 'A' + chr(ord(e) - ord('Z') + ord('A'))
            if not e.isupper():
                e = 'A' + chr(ord(e) - ord('Z') + ord('A'))
            # print("---", e)
            for metric in metrics:
                worksheet.conditional_format('{}{}:{}{}'.format(s,row+2,e,row+2), {
                    'type':     'cell',
                    'criteria': '==',
                    'value':    maxvals[metric][row],
                    'format':   format1,
                })
