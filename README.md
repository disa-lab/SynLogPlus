# Syn+: Improving the Templates Identified by Syntax-based Log Parsers

## Datasets Characteristics

| Software systems          | # Annotated Logs (Loghub-2.0) | # Templates  (Loghub-2.0) | # Templates (Loghub-2k) |
| ------------------------- | ------------------------- | --------------------- | ----------------------- |
| Hadoop                    | 179,993                   | 236                   | 114                     |
| OpenStack                 | 207,632                   | 48                    | 43                      |
| Zookeeper                 | 74,273                    | 89                    | 50                      |
| HPC                       | 429,987                   | 74                    | 46                      |
| Linux                     | 23,921                    | 338                   | 118                     |
| Mac                       | 100,314                   | 626                   | 341                     |
| Apache                    | 51,977                    | 29                    | 6                       |
| OpenSSH                   | 638,946                   | 38                    | 27                      |
| HealthApp                 | 212,394                   | 156                   | 75                      |
| Proxifier                 | 21,320                    | 11                    | 8                       |


## Repository Organization

```
├── 2k_dataset/ # The original Loghub-2k datasets
├── full_dataset/ # Loghub-2.0 datasets
├── benchmark/
│   ├── evaluation/ # Evaluation scripts for benchmark log parsers
│   ├── logparser/  # Benchmark log parsers (syntax-based)
│   ├── old_benchmark/
│   ├── LogPPT/     # contains the modified source code of LogPPT
│   ├── LLMParser/  # contains the modified source code of LLMParser
│   └── UniParser/  # contains the source code of implemented UniParser
├── result/
│   ├── ...... #
│   └── ...... # contains the output evaluation metric files and all parsed results
├── requirements.txt
└── README.MD
```

## Requirements

### System

Owing to the large scale of the benchmark datasets in the experiments, the
requirements of the benchmark of all log parsers are:

- At least 16GB memory
- At least 100GB storage
- GPU (for LogPPT, UniParser, and LLMParser)

### Dependencies

1. Python 3.10
2. Packages listed in requirements.txt


## Run evaluation

To evaluate a log parser, the following steps are taken:

1. The logs in the datasets are parsed by the log parser
2. The parsed logs are evaluated for accuracy

### Running the syntax-based log parsers

The 9 benchmark syntax-based log parsers can be run with their respective runner
scripts in the `benchmark/evaluation/` directory. Example commands are provided
below.  Replace `Drain` with other syntax-based log parsers.

```
pushd benchmark/evaluation/
python Drain_run.py             # For Loghub-2k datasets
python Drain_run.py -full       # For Loghub-2.0 datasets
```

### Running Syn+

Replace `Drain` with other syntax-based log parsers for it to be considered as
the grouping module.

```
pushd benchmark/evaluation/
python GeLL_run.py -g Drain -full
```

### Running semantic-based log parsers

Since these techniques are different from other syntax-based parsers and also
from each other, we seperate their environments from other log parsers.  Please
refer to the individual README files for UniParser, LogPPT, and LLMParser.


### Evaluating the log parsing results

When we have the parsed logs of the log parsers, we can evaluate the accuracy
across 4 accuracy metrics GA, PA, FGA, and FTA with the help of our evaluator
script. The runner scripts store the parsed logs in the `results/` directory.
The evaluator script accepts the directory path of the parsed logs, as shown in
the command below.  The evaluator scripts prints out the 4 accuracy metrics for
each dataset along with the average accuracies.  The script also stores the
evaluation results on a CSV file `results.csv` inside the directory provied.

```
pushd benchmark/
python evaluator.py --dirpath ../result/result_Drain_full/ --use_full
```


## 🔥 Citation

If you use our benchmark or datasets for research, please cite the following papers:

- Zhihan Jiang, Jinyang Liu, Junjie Huang, Yichen Li, Yintong Huo, Jiazhen Gu, Zhuangbin Chen, Jieming Zhu, Michael R. Lyu. [A Large-scale Evaluation for Log Parsing Techniques: How Far are We?](https://arxiv.org/abs/2308.10828) ISSTA, 2024.

- Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. [Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics](https://arxiv.org/abs/2008.06448). ISSRE, 2023.

In addition, if you use the souce code of our benchmark for research, please also cite the following two papers:

- Khan Zanis Ali, Shin Donghwan, Bianculli Domenico, Briand Lionel. [Guidelines for Assessing the Accuracy of Log Message Template Identification Techniques.](https://dl.acm.org/doi/abs/10.1145/3510003.3510101) ICSE, 2022.

- Jieming Zhu, Shilin He, Jinyang Liu, Pinjia He, Qi Xie, Zibin Zheng, Michael R. Lyu. [Tools and Benchmarks for Automated Log Parsing.](https://arxiv.org/abs/1811.03509) ICSE, 2019.
