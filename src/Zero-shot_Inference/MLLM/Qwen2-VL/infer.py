from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import sys
import json
import re
import os
from torch import nn
import csv
import argparse
from tqdm import tqdm
from torchvision import io
from PIL import Image
import av
import numpy as np
from decord import VideoReader, cpu
import av
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

def load_video(video_path, max_frames_num=4, fps=1, force_sample=False):

    container = av.open(video_path)

    total_frames = container.streams.video[0].frames

    indices = np.linspace(0, total_frames - 1, max_frames_num).astype(int)

    video_fps = container.streams.video[0].average_rate

    frame_time = [i / video_fps for i in indices]

    video_time = total_frames / video_fps

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

    frame_paths = []
    output_dir = './image'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, frame in enumerate(clip):
        frame_image = frame  # 直接使用 NumPy 数组，已经是 HxWxC 格式
        frame_pil = Image.fromarray(frame_image)
        frame_filename = os.path.join(output_dir, f'frame{i+1}.jpg')
        frame_pil.save(frame_filename)
        frame_paths.append(frame_filename)

    return frame_paths, frame_time, video_time

def evaluate_videos(args, model, processor, json_file_path, video_path, labels_dict, logger):
    
    data = load_json_data(json_file_path)

    results = []

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

        frame_paths, frame_time, video_time = load_video(modal_path, max_frames_num=4)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": [frame for frame in frame_paths],
                        "max_pixels": 360 * 420,
                        "fps": 1,
                    },
                    {"type": "text", "text": f"{instruction}"},
                ],
            }
        ] 
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if isinstance(output, list):
            output = output[0]
        
        logger.info(f"id:{id}, video_name:{video_name}, task: {args.task}, pred: {output}, true: {true_label}")
        
        results.append((video_name, args.task, output, true_label))

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    save_file_name = '_'.join([args.dataset, args.task, 'results.csv']) 
    save_results_to_csv(results, os.path.join(args.results_path, save_file_name))


def inference(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype="bfloat16", 
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(args.base_model_path)
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
        evaluate_videos(args, model, processor, json_file_path, video_path, labels_dict, logger)

if __name__ == "__main__":

    args = parse_arguments()
    inference(args)

