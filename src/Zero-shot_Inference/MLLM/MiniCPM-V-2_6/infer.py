import sys
import json
import os
import csv
from tqdm import tqdm
sys.path.append('./')
# from videollama2 import model_init, mm_infer
# from videollama2.utils import disable_torch_init
import torch 
import argparse
import logging

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
        "sentiment": ['neutral', 'positive', 'negative']
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
        'sentiment_regression': ['-1.0', '-0.8', '-0.6', '-0.4', '-0.2', '0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
        'sentiment': ['positive', 'negative'],  
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

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and attach to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_model_path', type=str, default='', help="LLM Path")
    
    parser.add_argument('--base_data_path', type=str, default='', help="BASE DATA PATH")
    
    parser.add_argument('--dataset', type=str, default='MIntRec', help="Dataset Name")

    parser.add_argument('--task', type=str, default='intent', help="Task Name")

    parser.add_argument('--results_path', type=str, default='results', help="Results Path")

    parser.add_argument('--device_ids', type=str, default='0', help="Device Ids")

    args = parser.parse_args()

    return args

def load_json_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def save_results_to_csv(results, csv_file_path):
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Video', 'Task', 'Pred', 'Label'])
        for result in results:
            csv_writer.writerow(result)

def evaluate_videos(args, model, processor, tokenizer, json_file_path, video_path, labels_dict, logger):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data = load_json_data(json_file_path)

    results = []
    wrong_id = []

    for item in tqdm(data, desc=f"Predicting {args.task}"):
        
        video_name = item['video']
        logger.info(f"Starting inference for video: {video_name}")

        modal_path = os.path.join(video_path, video_name)
        id= item['id']
        context = item['conversations'][0]['value']
        true_label = item['conversations'][1]['value']
        task_labels = labels_dict[args.task]

        modify_task = ' '.join(args.task.split('_'))

        instruction = f"You are presented with a video in which the speaker says: {context}."
        instruction += f" Based on the text, video, and audio content, what is the {modify_task} of this speaker?\n"
        instruction += f"The candidate labels for {modify_task} are: [{', '.join(task_labels)}]."
        instruction += f"Respond in the format: '{modify_task}: [label]'. "
        instruction += f' Only one label should be provided.'

        frames = processor['video'](modal_path)
        
        output = mm_infer(
            frames, 
            instruct=instruction,
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            modal='video'
        )
            
        logger.info(f"id:{id}, video_name:{video_name}, task: {args.task}, pred: {output}, true: {true_label}")
        

        results.append((video_name, args.task, output, true_label))

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    save_file_name = '_'.join([args.dataset, args.task, 'results.csv']) 
    save_results_to_csv(results, os.path.join(args.results_path, save_file_name))
    
def inference(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    video_path = os.path.join(args.base_data_path, args.dataset, 'video')
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    file_name = 'test_' + args.task + '.json'
    json_file_path = os.path.join(args.base_data_path, args.dataset, file_name)
    labels_dict = labels_map[args.dataset]

    model, processor, tokenizer = model_init(args.base_model_path)

    model_name = os.path.basename(args.base_model_path)
    log_file = os.path.join(args.results_path, f'inference_log_{model_name}_{args.dataset}.txt')
    logger = setup_logger(log_file)

    evaluate_videos(args, model, processor, tokenizer, json_file_path, video_path, labels_dict, logger)

if __name__ == "__main__":

    args = parse_arguments()
    inference(args)