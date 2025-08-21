import re
import os
import sys
import csv
import copy
import json
import math
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from old_benchmark.Drain_benchmark import benchmark_settings

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

class Logcluster:
    def __init__(self, logTemplate='', logIDL=None):
        self.logTemplate = logTemplate
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL

class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        if childD is None:
            childD = dict()
        self.childD = childD
        self.depth = depth
        self.digitOrtoken = digitOrtoken

class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, st=0.4, 
                 maxChild=100, rex=[], keep_para=False):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            depth : depth of all leaf nodes
            st : similarity threshold
            maxChild : max number of children of an internal node
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para
        self.delims = [ c for c in ' ,!@#$%^&(){}[]=-_:"' ]

    def hasNumbers(self, s):
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn, seq):
        retLogClust = None

        seqLen = len(seq)
        if seqLen not in rn.childD:
            return retLogClust

        parentn = rn.childD[seqLen]

        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break

            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else:
                return retLogClust
            currentDepth += 1

        logClustL = parentn.childD

        retLogClust = self.fastMatch(logClustL, seq)

        return retLogClust

    def addSeqToPrefixTree(self, rn, logClust):
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        for token in logClust.logTemplate:

            #Add current log cluster to the leaf node
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            #If token not matched in this layer of existing tree. 
            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if '<*>' in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']
                    else:
                        if len(parentn.childD)+1 < self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD)+1 == self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']
            
                else:
                    if '<*>' not in parentn.childD:
                        newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD['<*>']

            #If the token is matched
            else:
                parentn = parentn.childD[token]

            currentDepth += 1

    #seq1 is template
    def seqDist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue
            if token1 == token2:
                simTokens += 1 

        retVal = float(simTokens) / len(seq1)

        return retVal, numOfPar


    def fastMatch(self, logClustL, seq):
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim>maxSim or (curSim==maxSim and curNumOfPara>maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if maxSim >= self.st:
            retLogClust = maxClust  

        return retLogClust

    def getTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append('<*>')

            i += 1

        return retVal

    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        # import IPython; IPython.embed()
        for logClust in logClustL:
            template_str = ' '.join(logClust.logTemplate)
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1) 
        self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)


        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False, columns=["EventId", "EventTemplate", "Occurrences"])


    def printTree(self, node, dep):
        pStr = ''   
        for i in range(dep):
            pStr += '\t'

        if node.depth == 0:
            pStr += 'Root'
        elif node.depth == 1:
            pStr += '<' + str(node.digitOrtoken) + '>'
        else:
            pStr += node.digitOrtoken

        print(pStr)

        if node.depth == self.depth:
            return 1
        for child in node.childD:
            self.printTree(node.childD[child], dep+1)


    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node()
        logCluL = []

        self.load_data()

        count = 0
        for idx, line in tqdm(self.df_log.iterrows(),total=len(self.df_log), ascii=' >='):
            logID = line['LineId']
            logmessageL = self.preprocess(line['Content']).strip().split()
            # logmessageL = filter(lambda x: x != '', re.split('[\s=:,]', self.preprocess(line['Content'])))
            matchCluster = self.treeSearch(rootNode, logmessageL)

            #Match no existing log cluster
            if matchCluster is None:
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)

            #Add the new log message to the existing cluster
            else:
                newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                matchCluster.logIDL.append(logID)
                if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate): 
                    matchCluster.logTemplate = newTemplate

            count += 1
            # if count % 1000 == 0 or count == len(self.df_log):
            #     print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))


        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        log_messages = self.df_log['Content'].tolist()
        predictions = self.fix_templates(logCluL,log_messages)
        _df = pd.DataFrame(data=list(zip(log_messages,predictions)))
        _df.to_csv("{}/{}_structured.csv".format(self.savePath, self.logName),
                   header=['Content','EventTemplate'], index=False, quoting=csv.QUOTE_ALL)

        # self.outputResult(logCluL)

        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def word_is_variable(self, word):
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


    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe 
        """
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
        print("Total lines: ", len(logdf))
        return logdf


    def generate_logformat_regex(self, logformat):
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

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list

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
                elif self.word_is_variable(word):
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
        # template = [ "<*>" if is_number(word) or self.word_is_variable(word) else word for word in msg ]
        template = [ "<*>" if self.word_is_variable(word) else word for word in msg ]
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

    def fix_templates(self, logClustL, log_messages):
        predictions = [0] * len(log_messages)

        for logClust in tqdm(logClustL, ascii=' >='):
            template = logClust.logTemplate
            group_member_indices = [ x-1 for x in logClust.logIDL ]

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

            # XXX: use a subset of the member logs for efficiency?
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
