import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import os
import csv
import warnings  
from PIL import Image
from decord import VideoReader, cpu    
from tqdm import tqdm
import json
import signal
import time
import concurrent.futures
from modelscope import AutoTokenizer, AutoModelForCausalLM
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

    parser.add_argument('--merge_model_path', type=str, default='', help="MERGE MODEL PATH")
    
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

warnings.filterwarnings("ignore") 

def run_model(args, model, tokenizer, json_file_path, labels_dict, logger):

    data = load_json_data(json_file_path)
    results = []

    for item in tqdm(data, desc=f"Predicting {args.task}"):
        
        id= item['id']
        video_name = item['video']
        logger.info(f"Starting inference for video: {video_name}")

        context = item['conversations'][0]['value']
        true_label = item['conversations'][1]['value']
        task_labels = labels_dict[args.task]

        modify_task = ' '.join(args.task.split('_'))

        instruction = f"You are presented with a video in which the speaker says: {context}."
        instruction += f" Based on the text, video, and audio content, what is the {modify_task} of this speaker?\n"
        instruction += f"The candidate labels for {modify_task} are: [{', '.join(task_labels)}]."
        instruction += f"Respond in the format: '{modify_task}: [label]'. "
        instruction += f' Only one label should be provided.'

        model_name = os.path.basename(args.base_model_path)

        if model_name.startswith('Llama_3_8B'):

            messages = [
                {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": instruction}
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]

            output = tokenizer.decode(response, skip_special_tokens=True)

        elif model_name.startswith('Llama_3_1') or model_name.startswith('Llama_3_2'):

            messages = [
                {"role": "user", "content": instruction}
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)

            response = model.generate(**inputs, do_sample=True, max_new_tokens=256)
            output = tokenizer.batch_decode(response, skip_special_tokens=True)
            output = output[0].replace('\n','')

        elif model_name.startswith('Internlm'):

            response, history = model.chat(tokenizer, "hello", history=[])
            response, history = model.chat(tokenizer, instruction, history = history)
            output = response.replace('\n','').lower()

        elif model_name.startswith('Qwen2'):

            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": instruction}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
            generated_ids = model.generate(**model_inputs, max_new_tokens = 1024)
        
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)[0]

        if isinstance(output, list):
            output = output[0]
    
        logger.info(f"id: {id}, video_name: {video_name}, task: {args.task}, pred: {output}, true: {true_label}")
    
        results.append((video_name, args.task, output, true_label))

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    save_file_name = '_'.join([args.dataset, args.task, 'results.csv']) 
    save_results_to_csv(results, os.path.join(args.results_path, save_file_name))

def inference(args):

    model_path = args.base_model_path if args.merge_model_path == '' else args.merge_model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map = "auto", trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto", trust_remote_code = True, torch_dtype = torch.float16)
    model_name = os.path.basename(model_path)

    model = model.eval()

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
        
    file_name = 'test_' + args.task + '.json'
    json_file_path = os.path.join(args.base_data_path, args.dataset, file_name)
    
    labels_dict = labels_map[args.dataset]
    
    model_name = os.path.basename(args.base_model_path)
    log_file = os.path.join(args.results_path, f'inference_log_{model_name}_{args.dataset}.txt')
    logger = setup_logger(log_file)

    run_model(args, model, tokenizer, json_file_path, labels_dict, logger)


if __name__ == "__main__":

    args = parse_arguments()
    inference(args)
    