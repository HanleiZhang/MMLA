import json
import os
import pandas as pd
import re
import shutil
import argparse
import csv
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tqdm import tqdm


labels_map = {
    'MIntRec': {
        'intent': ['complain', 'praise', 'apologise', 'thank', 'criticize', 'agree', 'taunt', 'flaunt', 'joke',
                   'oppose', 'comfort', 'care', 'inform', 'advise', 'arrange', 'introduce', 'leave', 'prevent',
                   'greet', 'ask for help']
    },
    'MIntRec2.0': {
        'intent': ['acknowledge', 'advise', 'agree', 'apologise', 'arrange',
                   'ask for help', 'asking for opinions', 'care', 'comfort', 'complain',
                   'confirm', 'criticize', 'doubt', 'emphasize', 'explain',
                   'flaunt', 'greet', 'inform', 'introduce', 'invite',
                   'joke', 'leave', 'oppose', 'plan', 'praise',
                   'prevent', 'refuse', 'taunt', 'thank', 'warn']
    },
    "MELD": {
        "emotion": ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'anger', 'disgust'],
    },
    "MELD-DA": {
        "dialogue_act": ['greeting', 'question', 'answer', 'statement-opinion', 'statement-non-opinion', 'apology',
                         'command', 'agreement', 'disagreement', 'acknowledge', 'backchannel', 'others']
    },
    "IEMOCAP": {
        'emotion': ['angry', 'happy', 'sad', 'neutral', 'frustrated', 'excited']
    },
    "IEMOCAP-DA": {
        'dialogue_act': ['greeting', 'question', 'answer', 'statement-opinion', 'statement-non-opinion', 'apology',
                         'command', 'agreement', 'disagreement', 'acknowledge', 'backchannel', 'others']
    },
    'Ch-sims': {
        'sentiment': ['neutral', 'positive', 'negative'],
    },
    'UR-FUNNY': {
        'speaking_style': ['humorous', 'serious']
    },
    'MUStARD': {
        'speaking_style': ['sincere', 'sarcastic']
    },
    'MOSI': {
        'sentiment': ['positive', 'negative']
    },
    'AnnoMi-therapist':{
        'communication_behavior':['question', 'therapist_input', 'reflection', 'other']
    },
    'AnnoMi-client':{
        'communication_behavior':['neutral', 'change', 'sustain']
    },
}

task_map = {
    'MIntRec': ['intent'],
    'MIntRec2.0': ['intent'],
    'MELD': ['emotion'],
    'MELD-DA': ['dialogue_act'],
    'Ch-sims': ['sentiment'],
    'IEMOCAP-DA': ['dialogue_act'],
    'MUStARD': ['speaking_style'],
    'MOSI': ['sentiment'],
    'IEMOCAP': ['emotion'],
    'UR-FUNNY':["speaking_style"],
    'AnnoMi-therapist':['communication_behavior'],
    'AnnoMi-client':['communication_behavior'],
}


def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='MIntRec', help="Dataset Name")

    parser.add_argument('--task', type=str, default='intent', help="TASK Name")

    parser.add_argument('--model', type=str, default='Qwen2-VL-72B-Instruct', help="MODEL Name")

    parser.add_argument('--results_path', type=str, default='', help="Results Path")

    parser.add_argument('--timestamp', type=str, default='None', help="timestamp")

    parser.add_argument('--learning_rate', type=str, default='None', help="learning_rate")

    parser.add_argument('--warmup_ratio', type=str, default='None', help="warmup_ratio")

    parser.add_argument('--num_train_epochs', type=str, default='None', help="num_train_epochs")
    
    parser.add_argument('--lr_scheduler_type', type=str, default='None', help="lr_scheduler_type")


    args = parser.parse_args()

    return args

def cal_metrics(df):

    labels = []
    preds = []

    for i in range(len(df)):
        pred = str(df.loc[i,"pred_processed"]).lower()
        label = str(df.loc[i,"Label"]).lower()
        labels.append(label)
        preds.append(pred)

        print('pred: {}, label: {}'.format(pred, label))

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    prec = precision_score(labels, preds, average='macro')
    rec = recall_score(labels, preds, average='macro')

    weighted_f1 = f1_score(labels, preds, average='weighted')
    weighted_prec = precision_score(labels, preds, average='weighted')

    results_dict = {
        'ACC': round(acc * 100, 2),
        'F1': round(f1 * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall': round(rec * 100, 2),
        'WF1': round(weighted_f1 * 100, 2),
        'WP': round(weighted_prec * 100, 2)
    }

    return results_dict

def clean(df, dataset, task, labels_map):
    cur_labels = labels_map[dataset][task]
    processed = []

    for i in tqdm(range(len(df))):

        pred = str(df.loc[i, 'Pred']).lower()
        true_label = str(df.loc[i, 'Label']).lower()

        match_list = []
        for label in cur_labels:
            pattern = r'(?<!\w)' + re.escape(label.lower()) + r'(?!\w)'
            if re.search(pattern, pred):
                match_list.append(label.lower())
        
        if len(match_list) == 0:
            processed.append("unknown")
        elif len(match_list) == 1:
            processed.append(match_list[0])
        else:
            processed.append("multi")
        
    df["pred_processed"] = processed
    return df

def save_results(file_path, results_dict):

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        ori = []
        ori.append(results_dict.values())
        df1 = pd.DataFrame(ori, columns = list(results_dict.keys()))
        df1.to_csv(file_path, index = False)
    else:
        df1 = pd.read_csv(file_path)
        new = pd.DataFrame(results_dict, index=[1])
        df1 = pd.concat([df1, new], ignore_index=True)
        df1.to_csv(file_path,index=False)
    
    data_diagram = pd.read_csv(file_path)
    
    print('test_results', data_diagram)

def process_predictions(args):
    
    if args.model == 'MiniCPM-V-2_6':
        save_file_name = '_'.join([args.dataset, args.task, 'result.jsonl'])  

        labels = []
        predicts = []
        videos = []

        json_file_path = os.path.join(args.results_path, save_file_name)

        with open(json_file_path, "r") as f:

            for line in f.readlines():
                row = json.loads(line)
                predict = row['response'].lower()
                label = row['labels'].split(":")[1].strip().lower()
                video = row['videos'][0].split("/")[-1]

                videos.append(video)    
                labels.append(label)
                predicts.append(predict)
            
            return labels, predicts, videos

    else: 
        save_file_name = '_'.join([args.dataset, args.task, 'results.csv'])
        csv_file_path = os.path.join(args.results_path, save_file_name)

        labels = []
        predicts = []
        videos = []
        df = pd.read_csv(csv_file_path, sep=',')

        for index, row in df.iterrows():
            video = row['Video']
            label = row['Label'].split(':')[1].strip().lower()
            predict = row['Pred'].lower()

            videos.append(video)    
            labels.append(label)
            predicts.append(predict)

        return labels, predicts, videos

if __name__ == '__main__':

    args = parse_arguments()

    labels, predicts, videos = process_predictions(args)
    fieldnames = ['Video', 'Pred', 'Label']
    save_file_name = '_'.join([args.dataset, args.task, 'results.tsv'])
    output_path = os.path.join(args.results_path, save_file_name)

    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        for label, predict, video in zip(labels, predicts, videos):
            writer.writerow({
                'Video': video,
                'Pred': predict,
                'Label': label
            })

    df = pd.read_csv(output_path, sep = '\t')
    print('df: {}'.format(df.columns))

    df_clean = clean(df, args.dataset, task=args.task, labels_map=labels_map)
    print('df_clean: {}'.format(df_clean))
    results_dict = cal_metrics(df_clean)
    results_dict.update({
       'dataset': args.dataset,
        'task': args.task,
        'model': args.model,
        'timestamp': args.timestamp
    })
    save_file_name = 'results.csv'
    save_results(os.path.join(args.results_path, save_file_name), results_dict)

    
