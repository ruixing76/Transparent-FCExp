import os
import openai
import random
import json
import time
import logging
import argparse
from tqdm import tqdm
from utils import *
from annotation.annotation_config import *

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# random seed
random.seed(2023)


def openai_generate(input_text, api_func, tries=3, wait_time=1):
    output_text = ""
    for n in range(tries + 1):
        if n == tries:
            raise openai.APIError(f"Tried {tries} times.")
        try:
            output_text = api_func(input_text)
        except (openai.APIError, openai.APIConnectionError,
                openai.RateLimitError, openai.Timeout) as e:
            logging.warning(e)
            logging.warning(f"Retry after {wait_time}s. (Trail: {n + 1})")
            time.sleep(wait_time)
            continue
        break
    return output_text


def gpt35(input_text, model="gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{'role': 'user', 'content': input_text}],
        temperature=1
    )
    output_text = completion.choices[0].message.content
    return output_text


def gpt4(input_text, model="gpt-4-0613"):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{'role': 'user', 'content': input_text}],
        max_tokens=1000,
        temperature=1
    )
    output_text = completion.choices[0].message.content
    return output_text


def update_prompt(prompt, info_dict):
    for item in info_dict.items():
        prompt = prompt.replace(f'<{item[0]}>', item[1])
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(description="explanation generation executor")
    parser.add_argument(
        "-input_path",
        type=str,
        default="./data/TransExp_data/raw_dataset.json",
        help="The path to the input data. JSON files with 'uclaim' are machine selected evidences. ",
    )
    parser.add_argument(
        "-output_dir",
        type=str,
        default="./data/TransExp_data/generated_explanation/",
        help="The path to the generated explanation.",
    )
    parser.add_argument(
        "-model_name",
        type=str,
        default="gpt3",
        choices=MODEL_LIST,
        help="The model to use.",
    )
    parser.add_argument(
        "-mode",
        type=str,
        default="full",
        choices=MODES,
        help="set modes to core or full",
    )
    parser.add_argument(
        "-no_save",
        action="store_true",  # default is False, unless use this argument
        help="Prevent saving execution results.",
    )
    parser.add_argument(
        "-test",
        action="store_true",
        help="Enter test mode, only use 10 examples.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_path = args.input_path
    output_dir = args.output_dir
    model_name = args.model_name
    mode = args.mode
    no_save = args.no_save
    is_test = args.test
    print(
        f"reading data from:{args.input_path}, executing model:{args.model_name}, saving result to:{args.output_dir}")

    # create dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read input data file
    with open(input_path) as f:
        raw_dataset = json.load(f)

    # read claim chain mapping file
    with open('./data/TransExp_data/claim_chain_mapping.json') as f:
        claim_chain_mapping = json.load(f)

    if is_test:
        sampled_keys = random.sample(list(raw_dataset.keys()), 10)
        sample_dataset = {key: raw_dataset[key] for key in sampled_keys}
    else:
        sample_dataset = raw_dataset

    if mode == "core":
        gpt_prompt = gpt_prompt_core
        llama_prompt = llama_prompt_core
    elif mode == "full":
        gpt_prompt = gpt_prompt_full
        llama_prompt = llama_prompt_full
    else:
        raise AssertionError("mode should be core or full.")

    openai.organization = BUDGET_ORG
    openai.api_key = BUDGET_KEY

    # execute the model
    results = {}
    if model_name.startswith("gpt"):
        func = gpt4 if model_name == "gpt4" else gpt35
        for key, sample in tqdm(sample_dataset.items()):
            claim = sample['claim']
            veracity = sample['label']
            passage = sample['passage'].split('\n')
            chain = sample['chain'][1]
            reasons_prompt = ""
            if mode == "core":
                for i in chain:
                    reasons_prompt += f"Reason[{str(i)}]{passage[i].split(':')[1]}\n"
            elif mode == "full":
                for i, sen in enumerate(passage):
                    reasons_prompt += f"Reason[{str(i)}]{sen.split(':')[1]}\n"
            prompt = update_prompt(gpt_prompt,
                                   {'reasons': reasons_prompt, 'claim': claim, 'veracity': veracity})
            if mode == "full":
                chain = claim_chain_mapping[claim]['chain']
            # generate explanation
            output = {'generation': openai_generate(prompt, func, tries=3, wait_time=1),
                      'chain': chain,
                      'prompt': prompt}
            results[key] = output

    if model_name.startswith("llama2"):
        if model_name == "llama2-7b":
            model_path = "meta-llama/Llama-2-7b-chat-hf"
            torch_dtype = torch.float32
        else:
            model_path = "meta-llama/Llama-2-70b-chat-hf"
            torch_dtype = torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=1984, padding_side="right",
                                                  use_fast=False)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        for key, sample in tqdm(sample_dataset.items()):
            claim = sample['claim']
            veracity = sample['label']
            passage = sample['passage'].split('\n')
            chain = sample['chain'][1]
            reasons_prompt = ""
            if mode == "core":
                for i in chain:
                    reasons_prompt += f"Reason[{str(i)}]{passage[i].split(':')[1]}\n"
            elif mode == "full":
                for i, sen in enumerate(passage):
                    reasons_prompt += f"Reason[{str(i)}]{sen.split(':')[1]}\n"
            prompt = update_prompt(llama_prompt,
                                   {'reasons': reasons_prompt, 'claim': claim, 'veracity': veracity})
            sequences = pipeline(
                prompt,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=1000,
                temperature=1,
                return_full_text=False
            )
            if mode == "full":
                chain = claim_chain_mapping[claim]['chain']
            output = {'generation': sequences[0]['generated_text'],
                      'chain': chain,
                      'prompt': prompt}
            results[key] = output

    # save the results to output_path
    if len(results) != 0 and not no_save:
        if mode not in MODES:
            raise ValueError("mode should be core or full.")
        filename = f'{model_name}_{mode}_output.json'

        write_json(results, os.path.join(output_dir, filename))
        print(f"Execution results saved to {os.path.join(output_dir, f'{model_name}_{mode}_output.json')}")


if __name__ == '__main__':
    main()
