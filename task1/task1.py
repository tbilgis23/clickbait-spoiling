#!/usr/bin/env python3
import transformers
import json
from transformers import T5ForConditionalGeneration, AutoTokenizer
import argparse
import torch
import gc

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input', type=str, required=True)
    args.add_argument('--output', type=str, required=True)
    return args.parse_args()

def predict(inputs):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained("/model/clickbait_spoiling_classification")
    model = T5ForConditionalGeneration.from_pretrained("/model/clickbait_spoiling_classification").to(device)
    outputs = []
    for post in inputs:
        input_ids = tokenizer(
            f"multiclass classification: {post['postText'][0]} {' '.join(post['targetParagraphs'])} </s>",
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(device).input_ids
        output_text = tokenizer.decode(model.generate(input_ids)[0], skip_special_tokens=True)
        json_output = {'uuid': post['uuid'], 'spoilerType': output_text}
        outputs.append(json_output)
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