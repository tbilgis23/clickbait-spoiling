#!/usr/bin/env python3
import transformers
import json
from transformers import T5ForConditionalGeneration, AutoTokenizer
import argparse
import gc
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input', type=str, required=True)
    args.add_argument('--output', type=str, required=True)
    return args.parse_args()

def predict_multies(inputs):
    torch.cuda.empty_cache()
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained("/models/clickbait_spoiling_multi")
    model= T5ForConditionalGeneration.from_pretrained("/models/clickbait_spoiling_multi").to(device)
    outputs = []
    for post in inputs:
        input_ids = tokenizer(
                f"question: {post['postText'][0]} context: {' '.join(post['targetParagraphs'])} </s>",
                max_length=512,
                truncation=True,
                return_tensors='pt'
            ).to(device).input_ids
        output_text = tokenizer.decode(model.generate(input_ids)[0], skip_special_tokens=True)
        output_text = output_text.replace(" |", "")
        json_output = {'uuid': post['uuid'], 'spoiler': output_text}
        outputs.append(json_output)
    return outputs

def predict_phrase(inputs):
    torch.cuda.empty_cache()
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained("/models/clickbait_spoiling_phrase")
    model= T5ForConditionalGeneration.from_pretrained("/models/clickbait_spoiling_phrase").to(device)
    outputs = []
    for post in inputs:
        input_ids = tokenizer(
                f"question: {post['postText'][0]} context: {' '.join(post['targetParagraphs'])} </s>",
                max_length=512,
                truncation=True,
                return_tensors='pt'
            ).to(device).input_ids
        output_text = tokenizer.decode(model.generate(input_ids)[0], skip_special_tokens=True)
        json_output = {'uuid': post['uuid'], 'spoiler': output_text}
        outputs.append(json_output)
    return outputs

def predict_passage(inputs):
    torch.cuda.empty_cache()
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained("/models/clickbait_spoiling_passage")
    model= T5ForConditionalGeneration.from_pretrained("/models/clickbait_spoiling_passage").to(device)
    outputs = []
    for post in inputs:
        input_ids = tokenizer(
                f"question: {post['postText'][0]} context: {' '.join(post['targetParagraphs'])} </s>",
                max_length=512,
                truncation=True,
                return_tensors='pt'
            ).to(device).input_ids
        output_text = tokenizer.decode(model.generate(input_ids)[0], skip_special_tokens=True)
        json_output = {'uuid': post['uuid'], 'spoiler': output_text}
        outputs.append(json_output)
    return outputs

def predict(inputs):
    outputs = []
    multi_inputs = []
    passage_inputs = []
    phrase_inputs = []
    for post in inputs:
        if post['tags'][0] == "passage":
            passage_inputs.append(post)
        elif post['tags'][0] == "multi":
            multi_inputs.append(post)
        elif post['tags'][0] == "phrase":
            phrase_inputs.append(post)
    outputs += predict_multies(multi_inputs)
    outputs += predict_passage(passage_inputs)
    outputs += predict_phrase(phrase_inputs)
    return outputs


def main():
    args = get_args()
    input_file = args.input
    output_file = args.output
    with open(input_file, 'r') as inp, open(output_file, 'w') as out:
        inp = [json.loads(i) for i in inp]
        for output in predict(inp):
            out.write(json.dumps(output) + '\n')

if __name__ == '__main__':
    main()