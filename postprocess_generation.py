# postprocessing of generated explanation for annotation
# specifically:
# segment the explanation into sentences
# extract citation from each sentence
# find and mask the sentence that contains the citation in explanation
# save the dataset, this is the final dataset for annotation & experiment
import random

import spacy
from spacy.language import Language
import re
import os
import json
import argparse
from tqdm import tqdm
from utils import *

# random seed
random.seed(2023)


def flatten_with_digit(l):
    return [int(item) for sublist in l for item in sublist]


def split_to_sentences(input_text):
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("custom_sentence_segmenter", before="parser")
    doc = nlp(input_text)

    @Language.component("custom_sentence_segmenter")
    def custom_sentence_segmenter(doc):
        for token in doc[:-1]:
            # Check if a token resembles a citation pattern (e.g., "[1][2][3]")
            if token.text.endswith("]") and token.text.count("[") > 1:
                doc[token.i + 1].is_sent_start = False
        return doc

    sentence_list = [sent.text for sent in doc.sents]
    # if citation is separated into a single sentence,
    # connect it to the previous sentence and delete this sentence
    for i in range(len(sentence_list)):
        if sentence_list[i].startswith("[") and i > 0:
            sentence_list[i - 1] += sentence_list[i]
            sentence_list[i] = ""
    sentence_list = [sen for sen in sentence_list if sen != ""]
    return sentence_list


def extract_citation(sentence_list):
    citation_list = []
    for sen in sentence_list:
        pattern = r'\[(\d+(?:,\s*\d+)*)\]'
        try:
            citations = [int(each) for each in re.findall(pattern, sen)]
        except ValueError:
            citations = [-1]
        citation_list.append(citations)
    return citation_list


def locate_sen_with_cit(citation_list, tgt_cit):
    ans_sens = []
    for i, cits in enumerate(citation_list):
        if tgt_cit in cits:
            ans_sens.append(i)
    if len(ans_sens) == 0:
        ans_sens = [-2]
    return ans_sens


def parse_args():
    parser = argparse.ArgumentParser(description="postprocessing of dataset")
    parser.add_argument(
        "-raw_data_path",
        type=str,
        default="./data/TransExp_data/raw_dataset.json",
        help="The path to the raw dataset.",
    )
    parser.add_argument(
        "-input_dir",
        type=str,
        default="./data/TransExp_data/generated_explanation/",
        help="The path to the input data.",
    )
    parser.add_argument(
        "-output_dir",
        type=str,
        default="./data/TransExp_data",
        help="final_dataset",
    )
    parser.add_argument(
        "-model_name",
        type=str,
        default="gpt35",
        choices=MODEL_LIST,
        help="The model to evaluate.",
    )
    parser.add_argument(
        "-mode",
        type=str,
        default="full",
        choices=MODES,
        help="set modes to core or full",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    raw_data_path = args.raw_data_path
    input_dir = args.input_dir
    output_dir = args.output_dir
    model_name = args.model_name
    mode = args.mode

    random_mask = False

    # load data and generated results
    if model_name == 'all':
        running_list = MODEL_LIST
    else:
        running_list = [model_name]
    if mode == 'full':
        # under full setting, this is the raw data with machine extracted reasons
        raw_data_path = f"./data/TransExp_data/raw_dataset_uclaim_{model_name}.json"
    with open(raw_data_path, 'r') as f:
        dataset = json.load(f)
    # TODO replace claim with article ID, add text attribute of the claim
    with open("./data/claim_mask_mapping.json", 'r') as f:
        mask_mapping = json.load(f)
    for key, sample in dataset.items():
        if 'masked' in sample.keys():
            del sample['masked']
    # init dict
    for key, sample in dataset.items():
        sample['explanation'] = {}
        sample['masked_cit'] = {}
        sample['ans_sens'] = {}
        sample['gen_cit'] = {}
    for name in running_list:
        print(f"running postprocessing for {name}...")
        with open(os.path.join(input_dir, f'{name}_{mode}_output.json'), 'r') as f:
            generated_explanation = json.load(f)
        for key, sample in tqdm(dataset.items()):
            exe_sample = generated_explanation[key]
            seg_exp = split_to_sentences(exe_sample['generation'])
            cit_list = extract_citation(seg_exp)
            sample['explanation'][name] = seg_exp
            sample['gen_cit'][name] = cit_list
            if random_mask:
                sample['masked_cit'][name] = random.sample(list(cit_list), 1)[0]
            else:
                sample['masked_cit'][name] = mask_mapping[key]
            sample['ans_sens'][name] = locate_sen_with_cit(cit_list, sample['masked_cit'][name])

        write_json(dataset, os.path.join(output_dir, f'{name}_{mode}_data.json'))


if __name__ == '__main__':
    main()
