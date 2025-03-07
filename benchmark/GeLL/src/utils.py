import re
import os
import sys
import copy

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def tokenize_words(log, tokenizer, word_maxlen):
    indexed_tokens = []
    for word in log:  # tokenize each word
        tokenized_text = tokenizer.tokenize(word)
        index_new = tokenizer.convert_tokens_to_ids(tokenized_text)
        if len(index_new) > word_maxlen:
            index_new = index_new[:word_maxlen]
        else:
            index_new = index_new + (word_maxlen - len(index_new)) * [0]
        indexed_tokens = indexed_tokens + index_new
    tokens_id_tensor = torch.tensor([indexed_tokens])
    tokens_index = np.squeeze(tokens_id_tensor.numpy()).tolist()
    return tokens_index

def sim(seq1, seq2):
    long  = copy.copy(seq1) if len(seq1) > len(seq2) else copy.copy(seq2)
    short = copy.copy(seq2) if len(seq1) > len(seq2) else copy.copy(seq1)
    simTokens = 0
    for token in short:
        if token in long:
            simTokens+=1
    retVal = simTokens / len(long)
    return retVal

def is_pure_number(s):
    for func in (float, lambda x: int(x, 0)):
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

def _wordcomp(prediction, template, flag=False):
    _template = list(template)
    for i, _token in enumerate(_template):
        token = re.sub(r'\W+', '', _token)
        _template[i] = token
    for i, _token in enumerate(prediction):
        token = re.sub(r'\W+', '', _token)
        if token != '' and token not in _template:
            # print("---")
            # print(token, _template)
            # print(i, template[i] if i < len(template) else "")
            if not is_number(token) and i < len(template) and "<*>" not in template[i]:
                return False
            if i > len(template):
                return False
    return True

def wordcomp(prediction, template, flag=False):
    # print(prediction)
    # print(template)
    template = list(template)
    _f = False
    if " ".join(prediction) == 'The change in focus caused us to need to do a layout begin' and template[-1] == '<*>$Token':
        _f = True
    for word in template:
        if word not in prediction and "<*>" not in word and not is_number(re.sub(r'\W+','',word)):
            return False
    # if _f:
    #     print(prediction)
    #     print(template)
    #     flag = True
    def get_index_in_template(idx,word):
        try:
            return template.index(word,idx)
        except ValueError:
            return None
    indices = [] # [ get_index_in_template(idx,word) for idx,word in enumerate(prediction) ]
    last_idx = None
    for idx,word in enumerate(prediction):
        if is_number(re.sub(r'\W+','',word)):
            indices.append(None)
            continue
        matched_index = get_index_in_template(last_idx+1 if last_idx else 0,word)
        last_idx = matched_index if matched_index else last_idx
        indices.append(matched_index)

    prev,next = None,None
    for i,idx in enumerate(indices):
        if idx is not None:
            prev = idx
        else:
            if is_number(prediction[i]):
                continue
            if prev is None:
                if "<*>" not in template[0]:
                    if flag:
                        print("e1")
                    return False
                else:
                    continue
            if i==len(indices)-1:
                # if flag:
                #     print("e2", template[-1])
                return "<*>" in template[-1]
            if any(char.isdigit() for char in prediction[i]):
                continue
            l = [ x for x in indices[i:] if x ]
            if prediction[i][-1]=='=' or prediction[i].strip()=='is' or prediction[i+1].strip()=='=':
                fl = True
                if len(l) > 0:
                    m = list(range(prev+1,l[0]))
                    if len(m)>0 and "<*>" in template[m[0]]:
                        fl = False
                if fl and "0.0" not in prediction[i]:
                    if flag:
                        print(indices)
                        print("e3", i, l, prediction[i:i+2])
                    return False
            if len(l) == 0:
                if "<*>" not in template[-1]:
                    if flag:
                        print("e4", i)
                    return False
                else:
                    continue
            next = l[0]
            if prev>=next:
                print(prediction)
                print(template)
                print(indices)
                print(i)
                print(prev, next)
                exit()
            assert prev<next, f"{prediction},{template},{i},{prev},{next}"
            for j in range(prev+1,next):
                if "<*>" not in template[j]:
                    if flag:
                        print("e5", j, prev,next)
                    return False

    # print(prediction)
    # print(template)
    # print(indices)
    # exit()

    return True

def has_consecutive_variable(log,key):
    def find_continuous_subsequences(nums):
        subsequences = []
        current_subsequence = []
        for i in range(len(nums)):
            if i > 0 and nums[i] != nums[i - 1] + 1:
                if len(current_subsequence) > 1:
                    subsequences.append(current_subsequence)
                current_subsequence = []
            current_subsequence.append(nums[i])
        if len(current_subsequence) > 1:
            subsequences.append(current_subsequence)
        return subsequences
    def convert_to_regex_pattern(string):
        regex_pattern = ''
        for char in string:
            if char.isalpha():
                regex_pattern += r'[a-zA-Z]'
            elif char.isdigit():
                regex_pattern += r'-?\d*'
            else:
                if char != '-':
                    regex_pattern += re.escape(char)
        return regex_pattern

    long  = copy.copy(log) if len(log)>len(key) else copy.copy(key)
    short = copy.copy(key) if len(log)>len(key) else copy.copy(log)
    long, short = list(long), list(short)
    save=[]
    for word in long:
        if word not in short:
           save.append(long.index(word))
    consecutive=find_continuous_subsequences(save)
    for i in range(len(consecutive)):
        start=consecutive[i][0]
        end=consecutive[i][len(consecutive[i])-1]
        comppare=0
        for word in long[start:end+1]:
            if comppare == 0:
                regex = convert_to_regex_pattern(word)
                comppare = 1
            else:
                if not re.findall(regex, word):
                    return False
        long[start:end + 1] = ['<*>']
    if len(long)==len(short):
        return True
    else:
        return False


def match_with_existing_groups(log_groups, logsplit, dataset, log_id):

    def get_templ(template, logsplit):
        template = list(template)
        for i, token in enumerate(template):
            if token not in logsplit or token in ['false','true'] or exclude_digits(token):
                template[i] = "<*>"
        # from itertools import groupby
        # template = [key for key, _group in groupby(template)]
        return tuple(template)

    candidates = []
    keys = sorted(log_groups.keys(), key=lambda x: len(x))
    for templ in keys:
        flag = False # if 'jetty' in "".join(logsplit) and 'jetty' in "".join(templ):
        if log_id == 1 and 'acquire' in templ:
            flag = True
        if wordcomp(logsplit, templ, flag):
            _templ = get_templ(templ, logsplit)
            if set(_templ) == {'<*>'}:
                if flag:
                    print("3----")
                    print(logsplit)
                    print(templ)
                    print("3----")
                continue
            # if not has_consecutive_variable(logsplit, _templ):
            #     if flag:
            #         print("4----")
            #         print(logsplit)
            #         print(_templ)
            #         print(templ)
            #         print("4----")
            #     continue
            if flag:
                print(templ)
                print(_templ)
            candidates.append([templ, _templ])
        else:
            if flag:
                print("2----")
                print(logsplit)
                print(templ)
                print("2----")

    if len(candidates) >= 1:
        sim_list=[]
        for cand in candidates:
            sim_list.append(sim(cand[1], logsplit))
        maxsim = sim_list.index(max(sim_list))
        templ  = candidates[maxsim][0]
        _templ = candidates[maxsim][1]
        # XXX: should do this, but because of groundtruth impurity, we can't
        if _templ != templ:
            log_groups[_templ] = log_groups.pop(templ)
        # if log_id==2:
        #     print(candidates)
        #     print("_templ:", _templ)
        #     plg(log_groups)
        return _templ
    return None

def exclude_digits(string):
    pattern = r'\d'
    digits = re.findall(pattern, string)
    if len(digits)==0:
        return False
    return len(digits)/len(string) >= 0.3


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

def mark_as_vars_with_common_patterns(msg):
    patterns = [
        r'((?<=^)|(?<=\W))([A-Fa-f0-9]{2}:){5,11}[A-Fa-f0-9]{2}(?=(\W|$))',
        r'((?<=^)|(?<=\W))(\d{1,4}(-|/)\d{1,2}(-|/)\d{1,4})(?=(\W|$))',
        r'((?<=^)|(?<=\W))\d{4}(-)\d{4}(?=(\W|$))',
        r'((?<=^)|(?<=\W))[0-9a-zA-Z]+@([0-9a-zA-Z]+\.)+[0-9a-zA-Z]+(?=(\W|$))',
        r'((?<=^)|(?<=\W))\/?(?:[-0-9a-zA-Z]+\.){2,}[-0-9a-zA-Z]+(?::?:\d+)?(?=(\W|$))',
        r'((?<=^)|(?<=\W))[+-]?(\d+s(\d+\s?ms)?|\d+\s?ms)(?=(\W|$))',
        r'((?<=^)|(?<=\W))(\d+(\.\d+)?)\s?[kmgKMG]i?[bB]?((\/s)|(ytes))?(?=(\W|$))',
        r'((?<=^)|(?<=\W))(\d+(\.\d+)?)[KMG]Hz(?=(\W|$))',
        r'((?<=^)|(?<=\W))(\d+(\.\d+))k(?=(\W|$))',
        r'((?<=^)|(?<=\W))(\/[\d+\w+\-_\.\#\$]*[\/\.][\d+\w+\-_\.\#\$\/*]*)+(\sHTTPS?\/\d\.\d)?(?=(\W|$))',
        r'((?<=^)|(?<=\W))([a-zA-Z]\:[\/\\][\d+\w+\-_\.\#\$]*([\/\\\.][\d+\w+\-_\.\#\$\\\/*]*)?)(?=(\W|$))',
        # r'(?<=\w\=)[^ "]+',
        # r'(?<=(\="))[^"]+',
    ]

    for pat in patterns:
        msg = re.sub(pat, "<*>", msg)
    return msg

def plg(lg):
    # print("{\n" + "\n".join("{!r}: {!r},".format(k, v) for k, v in lg.items()) + "\n}")
    pass


def update_templates(log_groups, log_messages):
    predictions = [0] * len(log_messages)
    delims = [ c for c in ' ,!@#$%^&(){}[]=-_:"' ]

    def remove_extra_spaces(msg, template, pflag=False):
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
        # exit()
        if new_template != template:
            print(template)
            print(new_template)
        return new_template

    def subvars(template):
        for _ in range(3):
            template = re.sub(r'\w+_<\*>', "<*>", template)
            template = re.sub(r'<\*>_\w+', "<*>", template)
            template = re.sub(r'<\*>\%(\W)', "<*>\\1", template)
            template = re.sub(r'(<\*>[ @:$_/]?)+<\*>', "<*>", template)
            template = re.sub(r'(<\*>, ?)+<\*>', "<*>", template)
            template = re.sub(r'(<\*>\+)+<\*>', "<*>", template)
            template = re.sub(r'(@<\*> )+@<\*>', "@<*>", template)
            template = re.sub(r'(<\*>#+)+<\*>', "<*>", template)
        return template

    def get_template_from_msgs(msg1,msg2,templ,flag=False, pflag=False):
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
            if is_number(word) or word.lower() in ['false','true','root','null'] or prev_word in ['=','is','are']:
                if pflag:
                    print("a1",word)
                template.append('<*>')
                continue
            if word in delims or ''.join(list(set(word)))=='.':
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
        template = subvars(template)
        template = remove_extra_spaces("".join(msg1), template, pflag)

        return template

    def fix_template(msg,templ, pflag=False):
        if pflag:
            print('f', msg)
        template = [ "<*>" if is_number(word) or word_is_variable(word) else word for word in msg ]
        template = "".join(template)
        template = subvars(template)
        template = remove_extra_spaces("".join(msg), template, pflag)
        return template

    def _helper(msg, pflag=False):
        msg = mark_as_vars_with_common_patterns(msg)
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
        return new_msgsplit

    for template,group_member_indices in tqdm(log_groups.items(),total=len(log_groups.keys())):
        pflag = False
        # if 1910 in group_member_indices:
        #     pflag = True
        msgs = [
            list(filter(None, _helper(log_messages[iii], pflag)))
            for iii in group_member_indices[:2]
        ]
        if len(group_member_indices) == 1:
            idx = group_member_indices[0]
            predictions[idx] = " ".join(template).strip()
            _template = fix_template(msgs[0], " ".join(template).strip(), pflag)
            predictions[idx] = _template
            continue
        _template = get_template_from_msgs(msgs[0],msgs[1],template, False, pflag)
        for idx in group_member_indices:
            predictions[idx] = _template

    return predictions


def wordsplit(log,dataset,regx=None,regx_use=False,granular=False):
    if granular:
        tokens = re.split(r'([\s,!@#$%^&(){}\[\]\-=_:])', log)
        tokens = [ token for token in tokens if token ]
        return tokens

    if dataset == 'HealthApp':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
        log = re.sub('#', ' ', log)
    if dataset == 'Android':
        log = re.sub('\(', '( ', log)
        log = re.sub('\)', ') ', log)
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'HPC':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Hadoop':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)

    if dataset == 'OpenSSH':
        log = re.sub('=', '= ', log)
        log = re.sub(':', ': ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Thunderbird':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Windows':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub('\[', '[ ', log)
        log = re.sub(']', '] ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Zookeeper':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Mac':
        log = re.sub('\[', '[ ', log)
        log = re.sub(']', '] ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'BGL':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Proxifier':
        log = re.sub('\(.*?\)', '', log)
        log = re.sub(':', ' ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Linux':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if regx_use == True:
        for ree in regx:
            log = re.sub(ree, '<*>', log)
    log = re.split(' +', log)
    return log
