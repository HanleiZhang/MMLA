import json
import os
import csv
import argparse
import torch
import av
import numpy as np
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
import copy
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
        'sentiment': ['positive', 'negative'],
    },
    "AnnoMi-therapist":{
        "communication_behavior":['question', 'therapist_input', 'reflection', 'other']
    },
    "AnnoMi-client":{
        "communication_behavior":['neutral', 'change', 'sustain']
    },
}

task_map = {
    'MIntRec': ['intent'],
    'MIntRec2.0': ['intent'],
    'MELD': ['emotion'],
    'MELD-DA': ['dialogue_act'],
    'Ch-sims': ['sentiment'],
    'MOSI': ['sentiment'],
    'IEMOCAP': ['emotion'],
    'IEMOCAP-DA':['dialogue_act'],
    'UR-FUNNY':["speaking_style"],
    "AnnoMi-therapist":["communication_behavior"],
    "AnnoMi-client":["communication_behavior"],
    'MUStARD':['speaking_style'],
}

# Setting up logger
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

    parser.add_argument('--base_data_path', type=str, default='/Datasets', help="BASE DATA PATH")
    
    parser.add_argument('--lora_path', type=str, default='None', help="MERGE MODEL PATH")
    
    parser.add_argument('--dataset', type=str, default='MIntRec', help="Dataset Name")

    parser.add_argument('--task', type=str, default='intent', help="Task Name")

    parser.add_argument('--results_path', type=str, default='results', help="Results Path")

    parser.add_argument('--device_ids', type=str, default='0', help="Device Ids")

    parser.add_argument('--run_name', type=str, default='run_name', help="Run Name")

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

def load_video(video_path):

    container = av.open(video_path)
    
    total_frames = container.streams.video[0].frames
    
    indices = np.linspace(0, total_frames - 1,  4).astype(int)
    
    def read_video_pyav(container, indices):
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
    clip = read_video_pyav(container, indices)
    
    return clip

def evaluate_videos(args, model, processor, json_file_path, video_path, tokenizer, labels_dict, logger):

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {model.device}")
    data = load_json_data(json_file_path)
    results = []

    for item in tqdm(data, desc=f"{args.dataset}-{args.task}"):
    
        video_name = item['video']

        modal_path = os.path.join(video_path, video_name)

        id= item['id']
        context = item['conversations'][0]['value']
        true_label = item['conversations'][1]['value']
        task_labels = labels_dict[args.task]

        modify_task = ' '.join(args.task.split('_'))
        instruction = (
            f"You are presented with a video in which the speaker says: {context}."
            f" Based on the text, video, and audio content, what is the {modify_task} of this speaker?\n"
            f"The candidate labels for {modify_task} are: [{', '.join(task_labels)}]."
            f"Respond in the format: '{modify_task}: [label]'. "
            f"Only one label should be provided."
        )

        video_frames = load_video(modal_path)
        video_frames = processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(device).bfloat16()
        video_frames = [video_frames]

        conv_template = "qwen_1_5"
        question = DEFAULT_IMAGE_TOKEN + f"{instruction}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        with torch.no_grad():
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            cont = model.generate(
                input_ids,
                images=video_frames,
                modalities=["video"],
                do_sample=False,
                max_new_tokens=1024,
            )

        output = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        
        logger.info(f"id:{id}, video_name: {video_name}, task: {args.task}, pred: {output}, true: {true_label}")
        results.append((video_name, args.task, output, true_label))

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    save_file_name = '_'.join([args.dataset, args.task, 'results.csv']) 
    save_results_to_csv(results, os.path.join(args.results_path, save_file_name))

def inference(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

    if args.lora_path != '':
        model_name = get_model_name_from_path(args.run_name)
        tokenizer, model, processor, max_length = load_pretrained_model(
            args.lora_path,
            args.base_model_path, 
            model_name, 
            torch_dtype="bfloat16", 
            device_map='auto',
            attn_implementation="flash_attention_2"
        )
    else:
        tokenizer, model, processor, max_length = load_pretrained_model(
            args.base_model_path, 
            None, 
            "llava_qwen", 
            torch_dtype="bfloat16", 
            device_map='auto',
        )

    model.eval()
    video_path = os.path.join(args.base_data_path, args.dataset, 'video')
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    file_name = 'test_' + args.task + '.json'
    json_file_path = os.path.join(args.base_data_path, args.dataset, file_name)

    labels_dict = labels_map[args.dataset]

    model_name = os.path.basename(args.base_model_path)
    log_file = os.path.join(args.results_path, f'inference_log_{model_name}_{args.dataset}.txt')
    logger = setup_logger(log_file)

    with torch.no_grad():
        evaluate_videos(args, model, processor, json_file_path, video_path, tokenizer, labels_dict, logger)

if __name__ == "__main__":

    args = parse_arguments()
    inference(args)