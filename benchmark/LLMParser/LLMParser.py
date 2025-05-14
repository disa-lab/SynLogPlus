import os
import torch
import sys
import argparse

# import datasets
from transformers import set_seed
from datetime import datetime
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from openprompt.pipeline_base import PromptForGeneration
from openprompt.prompts.generation_verbalizer import GenerationVerbalizer

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--train_percentage", type=str, default="50_2000h")
parser.add_argument("--model", type=str, default="flan-t5-base")
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--full", action='store_true')
parser.add_argument(
    "--systems",
    type=str,
    default="Proxifier,Linux,Apache,Zookeeper,Mac,Hadoop,OpenStack,HealthApp,HPC,OpenSSH,BGL,HDFS,Spark,Thunderbird,Android,Windows",
)
parser.add_argument("--validation", type=str, default="validation")
args = parser.parse_args()


use_cuda = True
seed = 41
set_seed(seed)
batch_size = args.batch_size
lr = args.learning_rate
num_epochs = args.num_epochs

if args.validation == "validation":
    validation = True
else:
    validation = False


model_name = "t5"
pretrainedmodel_path = "./LLMs/{}/".format(args.model)
train_percentage=args.train_percentage

validation = False
test = True

dtype = 'full' if args.full else '2k'
suffix = '' if args.full else '_corrected'

logdir = 'full_dataset' if dtype== 'full' else 'logs'

def prepare_data(project,precentage):
    dataset_path = f"./training_samples_{dtype}/{precentage}/{project}/train.json"
    raw_dataset = pd.read_json(dataset_path)
    # raw_dataset = raw_dataset.head(3)
    raw_dataset = raw_dataset.drop(columns=['instruction'])
    raw_dataset = raw_dataset.map( str)  # must convert to string or else will hit error
    new_column_names = {'input': 'Content', 'output': 'EventTemplate', 'Template': 'EventTemplate'}
    raw_dataset = raw_dataset.replace( r"\s+", " ", regex=True)
    raw_dataset.rename(columns=new_column_names, inplace=True)

    # validation
    validation_dataset = pd.read_csv(f"../../{logdir}/{project}/{project}_{dtype}.log_structured{suffix}.csv")
    # validation_dataset = validation_dataset.head(1000)
    validation_dataset = validation_dataset.map( str)  # must convert to string or else will hit error
    validation_dataset = validation_dataset.replace( r"\s+", " ", regex=True)
    validation_dataset.rename(columns=new_column_names, inplace=True)

    # test
    test_dataset = pd.read_csv(f"../../{logdir}/{project}/{project}_{dtype}.log_structured{suffix}.csv")
    # test_dataset = test_dataset.head(1000)
    test_dataset = test_dataset.map( str)  # must convert to string or else will hit error
    test_dataset.rename(columns=new_column_names, inplace=True)
    test_dataset = test_dataset.replace( r"\s+", " ", regex=True)

    train_val_test = {}
    train_val_test["train"] = Dataset.from_dict(raw_dataset)
    train_val_test["validation"] = Dataset.from_dict(validation_dataset)
    train_val_test["test"] = Dataset.from_dict(test_dataset)
    print(train_val_test)


    from openprompt.data_utils import InputExample

    dataset = {}
    for split in ["train", "validation", "test"]:
        dataset[split] = []
        for data in train_val_test[split]:
            input_example = InputExample(
                text_a=data["Content"],
                meta={"EventTemplate": data["EventTemplate"]},
                label=0,
            )
            dataset[split].append(input_example)
    return dataset


project_list = args.systems.split(",")

if dtype == 'full':
    exclusion_list = ['Android','Windows','Spark','BGL','HDFS','Thunderbird']
    project_list = [ p for p in project_list if p not in exclusion_list ]

for project in project_list:
    # if project not in ['Apache']: continue

    start_time = datetime.now()
    dataset = prepare_data(project=project,precentage=train_percentage)
    print(
        "Project: {}, Seed : {} , Model : {} , Epoch : {} , Batch_size : {} , Learning Rate : {}".format(
            str(project),
            str(seed),
            pretrainedmodel_path,
            str(num_epochs),
            str(batch_size),
            str(lr),
        )
    )
    print(
        "\ntrain dataset : {} , validation dataset : {} , test dataset : {}".format(
            str(len(dataset["train"])),
            str(len(dataset["validation"])),
            str(len(dataset["test"])),
        )
    )
    # load plm
    from openprompt.plms import load_plm
    from openprompt.plms.seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper
    from openprompt.plms.lm import LMTokenizerWrapper
    from transformers import T5TokenizerFast

    new_words = ["<*>", "{", "}", "<", "\\"]
    plm, tokenizer, model_config, WrapperClass = load_plm(
        model_name, pretrainedmodel_path
    )
    tokenizer = T5TokenizerFast.from_pretrained(pretrainedmodel_path)
    tokenizer.add_tokens(new_tokens=new_words)

    from openprompt.prompts import ManualTemplate

    template_text = (
        'Parse the raw log to log template: {"placeholder":"text_a"}  {"mask"}'
    )
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

    # define the verbalizer
    from openprompt.data_utils.utils import InputExample
    from openprompt.prompts import GenerationVerbalizer

    label_words = {0: ["{'meta':'EventTemplate'}"]}
    myverbalizer = GenerationVerbalizer(
        tokenizer, classes=None, is_rule=True, label_words=label_words
    )

    # define prompt model for classification
    prompt_model = PromptForGeneration(
        plm=plm, template=mytemplate, tokenizer=tokenizer
    )
    if use_cuda:
        prompt_model = prompt_model.cuda()
    # DataLoader
    from openprompt import PromptDataLoader

    train_dataloader = PromptDataLoader(
        dataset=dataset["train"],
        template=mytemplate,
        verbalizer=myverbalizer,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        decoder_max_length=256,
        batch_size=batch_size,
        shuffle=True,
        teacher_forcing=True,
        predict_eos_token=True,
        # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
        truncate_method="tail",
    ).dataloader

    validation_dataloader = PromptDataLoader(
        dataset=dataset["validation"],
        template=mytemplate,
        verbalizer=myverbalizer,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        decoder_max_length=256,
        batch_size=batch_size,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
        truncate_method="tail",
    ).dataloader

    test_dataloader = PromptDataLoader(
        dataset=dataset["test"],
        template=mytemplate,
        verbalizer=myverbalizer,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        decoder_max_length=256,
        batch_size=batch_size,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
        truncate_method="tail",
    ).dataloader
    validation_dataloader_inputs_decode = []
    for step, inputs in enumerate(validation_dataloader.dataset):
        inputs_id = inputs["input_ids"]
        deco = tokenizer.decode(inputs_id, skip_special_tokens=True).replace(
            "Parse the raw log to log template: ", ""
        )
        validation_dataloader_inputs_decode.append(deco)

    test_dataloader_inputs_decode = []
    for step, inputs in enumerate(test_dataloader.dataset):
        # print(inputs)
        inputs_id = inputs["input_ids"]
        # print(tokenizer.decode(outputs, skip_special_tokens=True))
        deco = tokenizer.decode(inputs_id, skip_special_tokens=True).replace(
            "Parse the raw log to log template: ", ""
        )
        test_dataloader_inputs_decode.append(deco)

    from transformers import get_linear_schedule_with_warmup

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in prompt_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in prompt_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    from openprompt.utils.crossfit_metrics import evaluate as crossfit_evaluate
    import numpy as np
    import csv

    generation_arguments = {
        "max_length": 128,
    }

    def accuracy(string1, string2):
        string1 = string1.replace(" ", "")
        string2 = string2.replace(" ", "")

        if string1 == string2:
            return True
        else:
            return False

    def get_accruacy_over_list(prediction, groundtruth):
        if type(groundtruth) == list:
            if len(groundtruth) == 0:
                return 0
            return np.max([accuracy(prediction, gt) for gt in groundtruth])
        return accuracy(prediction, groundtruth)

    def acc(predictions, data, metric):
        if metric == "ACC":
            accs = []
            for prediction, dp in zip(predictions, data):
                accs.append(get_accruacy_over_list(prediction, dp))
            return np.mean(accs)

    def evaluate(prompt_model, dataloader, mode="test", diff_out_path=""):
        prompt_model.eval()
        predictions = []
        ground_truths = []
        csv_out = []
        res_out = []
        if mode == "test":
            inputs_decode = test_dataloader_inputs_decode
        elif mode == "validation":
            inputs_decode = validation_dataloader_inputs_decode
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(dataloader)):
                if use_cuda:
                    inputs = inputs.cuda()
                a, output_sentence = prompt_model.generate(
                    inputs, **generation_arguments
                )
                predictions.extend(output_sentence)
                ground_truths.extend(inputs["tgt_text"])
        assert len(predictions) == len(ground_truths), (
            len(predictions),
            len(ground_truths),
        )
        predictions = [prediction.strip() for prediction in predictions]
        ground_truths = [ground_truth.strip() for ground_truth in ground_truths]
        score = acc(predictions, ground_truths, metric="ACC")
        if mode =="test":
            for i in range(len(inputs_decode)):
                if not accuracy(predictions[i], ground_truths[i]):
                    csv_out.append([inputs_decode[i], predictions[i], ground_truths[i]])
            if not os.path.exists(diff_out_path):
                os.makedirs(diff_out_path)
            res_out.append(['Content','EventTemplate','GroundTruth'])
            for i in range(len(inputs_decode)):
                res_out.append([inputs_decode[i], predictions[i], ground_truths[i]])
            with open(diff_out_path + project + "_full.log_structured.csv", "w") as f:
                f = csv.writer(f)
                f.writerows(res_out)
            with open(diff_out_path + project + "_diff.csv", "w") as f:
                f = csv.writer(f)
                f.writerows(csv_out)
            with open(diff_out_path + project + "_result.txt", "w") as f:
                f.write(str(score))
        print("acc: {}".format(score))
        return predictions,ground_truths,score

    tot_loss = 0
    log_loss = 0
    best_val_acc = 0
    glb_step = 0
    actual_step = 0
    leave_training = False

    acc_traces = []
    tot_train_time = 0
    pbar_update_freq = 10

    output_dir = "./fine_tuned_model/{}/{}/{}/".format(args.model, project,train_percentage)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    des=project+str(train_percentage)
    pbar = tqdm(range(num_epochs), desc="Loss: -0.000")
    for epoch in pbar:
        # diff_out_path = "./output/{}/{}/".format(args.model, train_percentage)
        diff_out_path = "../../result/result_LLMParser_{}/".format(dtype)
        # train
        prompt_model.train()
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            loss = prompt_model(inputs)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        pbar.set_description(f"Loss: {tot_loss/(step+1):.3f}")
        if validation:
            if (epoch + 1) % 5 == 0 or (epoch <5):
                # validate
                print("\n\nepoch{}------------validate------------".format(epoch))
                predictions,ground_truths,val_acc = evaluate(
                    prompt_model,
                    validation_dataloader,
                    mode="validation",
                    diff_out_path=diff_out_path,
                )
                acc_traces.append(val_acc)
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    bestepoch = epoch
                    # save_dir = os.path.join(output_dir, 'bestepoch')
                    plm.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    # test
                    print("\n\nepoch{}------------test------------".format(epoch))
                    predictions,ground_truths,test_acc = evaluate(
                        prompt_model,
                        test_dataloader,
                        mode="test",
                        diff_out_path=diff_out_path,
                    )

    if test:
        print("\n\n------------test------------")
        predictions,ground_truths,test_acc = evaluate(
            prompt_model,
            test_dataloader,
            mode="test",
            diff_out_path=diff_out_path,
        )

    if not validation:
        plm.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    finish_time = datetime.now()
    duration = finish_time - start_time

    days = duration.days
    hours = duration.seconds // 3600
    minutes = (duration.seconds // 60) % 60
    seconds = duration.seconds % 60

    print(
        f"\n\nRunning: {days} days, {hours} hours, {minutes} mins, {seconds} sec."
    )
