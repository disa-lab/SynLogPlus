import re
import os
import sys
import csv
import copy
import json
import math
import argparse
import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from old_benchmark.Drain_benchmark import benchmark_settings



def load_data(log_format, log_file):
    headers, regex = generate_logformat_regex(log_format)
    return log_to_dataframe(log_file, regex, headers, log_format)

def log_to_dataframe(log_file, regex, headers, logformat):
    log_messages = []
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf


def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex

from logparser.Drain import LogParser as drain


def plg(lg):
    # print("{\n" + "\n".join("{!r}: {!r},".format(k, v) for k, v in lg.items()) + "\n}")
    pass


def is_pure_number(s):
    bases = [0]
    if len(s)>=6: bases.append(16)
    for b in bases:
        for func in (float, lambda x: int(x, b)):
            try:
                func(s)
                return True
            except ValueError:
                continue
    return False

def is_number(s):
    if is_pure_number(s) or all([ is_pure_number(_s) for _s in re.split(r'[\/]',s) ]):
        return True
    n_digits = 0
    n_alphas = 0
    for c in s:
        if c.isalpha():
            n_alphas+=1
        elif c.isdigit():
            n_digits+=1
    return n_digits > n_alphas

def word_is_variable(word):
    patterns = [
        r'(^|\W+)(\d){1,2}:(\d){1,2}(|:(\d){2,4})(\W+|$)',
        r'(^|\W)(\d{1,2}(-|/)\d{1,2}(-|/)\d{2,4})(\W|$)',
        # r'(^|\W)(?:[-0-9a-zA-Z]+\.)+[-0-9a-zA-Z]+(?::?:\d+)?',
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)',
        r'(^|\W)[-0-9a-zA-Z]+(?::?:\d+)',
        r'(|^)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(|$)',
        r'(^|\W)(0x)?[A-Fa-f0-9]{5,}(\W|$)',
    ]

    common_vars = ['false','true','root','null']
    if word.lower() in common_vars:
        return True

    for pat in patterns:
        if re.findall(pat,word):
            return True
    return False

class LogParser:
    def __init__(self, dataset):
        self.dataset = dataset
        self.delims = [ c for c in ' ,!@#$%^&(){}[]=-_:"' ]

    def read_logs(self, logdir, dataset, use_full=False):
        logdir = Path(logdir) if isinstance(logdir,str) else logdir
        if use_full:
            log_file = logdir / dataset / '{}_full.log'.format(dataset)
            ground_file = logdir / dataset / '{}_full.log_structured.csv'.format(dataset)
        else:
            log_file = logdir / dataset / '{}_2k.log'.format(dataset)
            ground_file = logdir / dataset / '{}_2k.log_structured_corrected.csv'.format(dataset)

        df = load_data(benchmark_settings[dataset]['log_format'], log_file)
        log_messages = df['Content'].map(str).tolist()
        # XXX: removing double-spaces
        log_messages = [re.sub(r'\s+', ' ', e.strip()) for e in log_messages]

        log_templates = pd.read_csv(ground_file,dtype=str)['EventTemplate'].map(str).tolist()
        return log_messages, log_templates

    def group_logs(self, grouped_csv_path):
        df = pd.read_csv(grouped_csv_path,dtype=str)
        log_messages = df['Content'].map(str).tolist()
        log_templates = df['EventTemplate'].map(str).tolist()
        log_groups = {}
        for idx,template in enumerate(log_templates):
            template_split = tuple(template.split())
            log_groups.setdefault(template_split, []).append(idx)
        return log_groups

    def anonymize_with_regex(self,msg):
        patterns = [
            r'((?<=^)|(?<=\W))([A-Fa-f0-9]{2}:){5,}[A-Fa-f0-9]{2}(?=(\W|$))',
            r'((?<=^)|(?<=\W))(\d{1,4}(-|/)\d{1,2}(-|/)\d{1,4})(?=(\W|$))',
            # Sat Jun 11 03:28:22 2005
            r'((?<=^)|(?<=\W))((Sat|Sun|Mon|Tue|Wed|Thu|Fri)\s)?((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s)(\d{,2}\s?)((\d{2}\:){2}\d{2}\s?)?([A-Z]{3}\s?)?(\d{4})?(?=(\W|$))',
            r'((?<=^)|(?<=\W))[0-9a-zA-Z]+@([0-9a-zA-Z]+\.)+[0-9a-zA-Z]+(?=(\W|$))',
            r'((?<=^)|(?<=\W))\/?(?:[-0-9a-zA-Z]+\.){2,}[-0-9a-zA-Z]+(?::?:\d+)?(?=(\W|$))',
            r'((?<=^)|(?<=\W))[+-]?(\d+s(\d+\s?ms)?|\d+\s?ms)(?=(\W|$))',
            r'((?<=^)|(?<=\W))(\d+(\.\d+)?)\s?[kmgKMG]i?[bB]?((\/s)|(ytes))?(?=(\W|$))',
            r'((?<=^)|(?<=\W))(\d+(\.\d+)?)[KMG]Hz(?=(\W|$))',
            # r'((?<=^)|(?<=\W))(\d+(\.\d+))k(?=(\W|$))',
            r'((?<=^)|(?<=\W))(\/[\d+\w+\-_\.\#\$]*[\/\.][\d+\w+\-_\.\#\$\/*]*)+(\sHTTPS?\/\d\.\d)?(?=(\W|$))',
            # r'((?<=^)|(?<=\W))(\/[\d+\w+\-_\.\#\$]+\/?)+(\/[\d+\w+\-_\#\$]+\/?)(\.[\d+\w+\-_\#\$\/*]+)?(?=(\W|$))',
            # r'((?<=^)|(?<=\W))HTTPS?\/\d+(\.\d+)?(?=(\W|$))',
            r'((?<=^)|(?<=\W))([a-zA-Z]\:[\/\\][\d+\w+\-_\.\#\$]*([\/\\\.][\d+\w+\-_\.\#\$\\\/*]*)?)(?=(\W|$))',
            # r'(?<=\w\=)[^ "]+',
            # r'(?<=(\="))[^"]+',
        ]

        for pat in patterns:
            msg = re.sub(pat, "<*>", msg)
        return msg

    def anonymize_numbers(self, log, pflag=False):
        template = [ "<*>" if is_number(token) else token for token in self.tokenize_log(log) ]
        template = "".join(template)
        if pflag:
            print("an", logsplit)
            print("an", template)
        return template

    def subvars(self, template):
        for _ in range(3):
            template = re.sub(r'\w+_<\*>', "<*>", template)
            template = re.sub(r'<\*>_\w+', "<*>", template)
            template = re.sub(r'<\*>\%(\W)', "<*>\\1", template)
            template = re.sub(r'(<\*>[ @:$_/]?)+<\*>', "<*>", template)
            template = re.sub(r'(<\*>, ?)+<\*>', "<*>", template)
            template = re.sub(r'(<\*>\+)+<\*>', "<*>", template)
            template = re.sub(r'(@<\*> )+@<\*>', "@<*>", template)
            template = re.sub(r'(<\*>#+)+<\*>', "<*>", template)
            template = re.sub(r'(?<=\<\*\> )\(\w+\)(?=(\W|$))', '(<*>)', template)
            # template = template.replace(' ()', ' (<*>)')
        return template

    def fix_spaces(self, msg, template, pflag=False):
        if pflag:
            print("r", msg)
            print("r", template)
        templsplit = template.split("<*>")
        # print(templsplit)
        for i, split in enumerate(templsplit):
            if split in msg: continue
            space_indices = [i for i, ltr in enumerate(split) if ltr == ' ']
            if pflag:
                print("r", space_indices)
                print("r", split)
            for j in space_indices:
                new_split = split[:j] + split[j+1:]
                if new_split in msg:
                    templsplit[i] = new_split
                    if pflag:
                        print("r", new_split)
                        print("r", templsplit)
                    continue
        new_template = "<*>".join(templsplit)
        if pflag:
            print("r", new_template)
        if new_template != template:
            print('Fixing spaces:')
            print('Wrong template:', template)
            print('Fixed template:', new_template)
        return new_template

    def extract_template(self, msg1,msg2,templ,flag=False, pflag=False):
        short = msg1 if len(msg1)<=len(msg2) else msg2
        long  = msg1 if short!=msg1 else msg2

        if pflag:
            print(short)
            print(long)
            print(templ)

        last_idx = None
        template = []

        def get_index_in_list(word, list, idx):
            try:
                return list.index(word,idx)
            except ValueError:
                return None

        prev_word = None
        for idx,word in enumerate(short):
            # if is_number(word) or word.lower() in ['false','true','root','null']: # or prev_word in ['=','is','are']:
            if word.lower() in ['false','true','root','null']: # or prev_word in ['=','is','are']:
                if pflag:
                    print("a1",word)
                template.append('<*>')
                continue
            if word in self.delims or ''.join(list(set(word)))=='.':
                if pflag:
                    print("a2",word)
                template.append(word)
                continue
            matched_index = get_index_in_list(word, long, last_idx+1 if last_idx is not None else 0)
            if matched_index is not None:
                if word in "".join(templ):
                    if pflag:
                        print("a3",word)
                    template.append(word)
                elif flag:
                    if pflag:
                        print("a4",word)
                    template.append('<*>')
                    continue
                elif idx+1<len(short) and (short[idx+1]=='=' or ''.join(list(set(short[idx+1])))=='.' ):
                    if pflag:
                        print("a5",word)
                    template.append(word)
                elif word_is_variable(word):
                    if pflag:
                        print("a6",word)
                    template.append("<*>")
                else:
                    if pflag:
                        print("a7",word)
                    template.append(word)
            else:
                if pflag:
                    print("a8",word)
                template.append('<*>')

        if pflag:
            print(template)
        template = "".join(template).strip()

        return template

    def refine_template(self, msg, templ, pflag=False):
        if pflag:
            print('f', msg)
        # template = [ "<*>" if is_number(word) or word_is_variable(word) else word for word in msg ]
        template = [ "<*>" if word_is_variable(word) else word for word in msg ]
        template = "".join(template)
        return template

    def tokenize_log(self, msg, pflag=False):
        if pflag:
            print("h", msg)
        msgsplit = re.split(r'(\.$|\.{5,}|[\s,;!@#$%^&(){}\[\]=_:"\+])', msg)
        new_msgsplit = []
        for split in msgsplit:
            if len(split) > 1 and split[-1] == '.':
                new_msgsplit.append(split[:-1])
                new_msgsplit.append(split[-1])
            else:
                new_msgsplit.append(split)
        if pflag:
            print("h", new_msgsplit)
        return new_msgsplit

    def post_process(self, log, template, pflag=False):
        template = self.subvars(template)
        # template = self.fix_spaces(log, template, pflag)
        # template = re.sub(r'\<\*\> sec$', '<*>', template)
        return template

    def is_a_match(self, log, template):
        pat = self.get_pattern_from_template(template)
        if pat.count('.*') > 10: return True
        # print(pat)
        matched = re.fullmatch(pat, log)
        return matched is not None

    def get_pattern_from_template(self, template):
        pat = re.sub(r'<\*>', 'esarhpodnarstot', template)
        escaped = re.escape(template)
        space_escaped = re.sub(r'\\\s+', "\\\s+", escaped)
        regpat = space_escaped.replace(r"<\*>", r".*")
        return regpat

    def fix_templates(self, log_groups, log_messages):
        predictions = [0] * len(log_messages)

        for template, group_member_indices in tqdm(
            log_groups.items(), total=len(log_groups.keys()),
            desc=self.dataset, ascii=' >-'
        ):
            pflag = False
            # if 863 in group_member_indices:
            #     pflag = True

            logs_in_this_group = [ log_messages[iii] for iii in group_member_indices ]
            # logs_in_this_group = list(set(logs_in_this_group))

            sampled_logs = []
            for iii in group_member_indices:
                if len(sampled_logs) > 2: continue
                if log_messages[iii] not in sampled_logs:
                    sampled_logs.append(log_messages[iii])

            anonymized_logs = [
                self.anonymize_with_regex(log) for log in sampled_logs
            ]
            anonymized_logs = [
                self.anonymize_numbers(log) for log in anonymized_logs
            ]
            tokenized_logs = [
                list(filter(None, self.tokenize_log(log, pflag)))
                for log in anonymized_logs
            ]

            if len(sampled_logs) == 1:
                extracted_template = self.refine_template(tokenized_logs[0], " ".join(template).strip(), pflag)
            else:
                extracted_template = self.extract_template(tokenized_logs[0],tokenized_logs[1], template, False, pflag)

            # extracted_template = self.post_process("".join(tokenized_logs[0]), extracted_template)

            for idx in group_member_indices:
                log = log_messages[idx]
                _template = extracted_template
                template_split = self.tokenize_log(_template.replace('<*>',''))
                for token in template_split:
                    if token not in log:
                        _template = _template.replace(token, '<*>')
                _template = self.post_process(log, _template)
                extracted_template = _template

            for idx in group_member_indices:
                predictions[idx] = extracted_template

            # pflag = False
            # for idx in group_member_indices:
            #     predictions[idx] = extracted_template
            #     log = log_messages[idx]
            #     if not self.is_a_match(log, extracted_template):
            #         if pflag:
            #             print("---")
            #             print(log)
            #             print(extracted_template)
            #         _template = extracted_template
            #         template_split = self.tokenize_log(_template.replace('<*>',''))
            #         for token in template_split:
            #             if token not in log:
            #                 _template = _template.replace(token, '<*>')
            #         if pflag:
            #             print(_template)
            #             print("---")
            #         predictions[idx] = _template


        return predictions

