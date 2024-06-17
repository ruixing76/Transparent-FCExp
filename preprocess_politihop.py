# preprocess the PolitiHop dataset
import os
import pickle
import pandas as pd
import ast
import tiktoken
from utils import *

train_path = "data/PolitiHop_data/politihop_train.tsv"
valid_path = "data/PolitiHop_data/politihop_valid.tsv"
test_path = "data/PolitiHop_data/politihop_test.tsv"

# using pandas
train_df = pd.read_csv(train_path, sep="\t")
valid_df = pd.read_csv(valid_path, sep="\t")
test_df = pd.read_csv(test_path, sep="\t")

train_df["annotated_evidence"] = train_df["annotated_evidence"].apply(eval)
valid_df["annotated_evidence"] = valid_df["annotated_evidence"].apply(eval)
test_df["annotated_evidence"] = test_df["annotated_evidence"].apply(eval)


# merge train and dev instances
def merge_dicts(dicts):
    merged_dict = {}
    for d in dicts:
        merged_dict.update(d)
    return merged_dict


aggregation = {"article_id": "first", 'author': "first",
               'ruling': "first", 'url_sentences': "first", 'relevant_text_url_sentences': "first",
               'politifact_label': "first", 'annotated_label': "first", 'urls': "first",
               'annotated_urls': "first", "annotated_evidence": merge_dicts}

m_train_df = train_df.groupby("statement").agg(aggregation).reset_index()
m_valid_df = valid_df.groupby("statement").agg(aggregation).reset_index()

# Preprocessing
original_data = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)
dataset = pd.concat([m_train_df, m_valid_df, test_df], axis=0, ignore_index=True)


def get_unique_chains(chains):
    """
    Since we don't have so many instances, we just remove duplicate chains in the evidence set
    """
    sets = set(map(tuple, list(chains.values())))
    unique_chains = {str(i): list(c) for i, c in enumerate(sets)}
    return unique_chains


def format_one_ins(sample):
    """ format sample as input to GPT, return an input_dict
        input_dict={"article_id":sample['article_id'],
                    "chain_dict":chain_dict,
                    "prompt_dict":prompt_dict}
        prompt_dict={'chain_id':<prompt of the chain>}
        each item in prompt_dict contains one standalone reasoning chain
    """

    CLAIM = f"Claim: {sample['statement']}"
    VERACITY = f"Veracity: {sample['annotated_label']}"

    # original reasoning chain dict {'chain_id': [list of evidence sentence number]}
    chains = sample["annotated_evidence"]
    # remove duplicate chains
    chain_dict = get_unique_chains(chains)
    # TEST original chain
    # unique_chains=chains

    ruling = ast.literal_eval(sample["ruling"])
    PASSAGE = "\n".join([str(i) + ": " + sen for i, sen in enumerate(ruling)])
    PRE = f"{gpt_prompt_core}\n{CLAIM}\n{VERACITY}\nReasoning chain:\n"
    prompt_dict = {}  # {'chain_id': prompt for this chain}

    for k, v in chain_dict.items():
        CHAIN = f"chain {k}:\n"
        new_v = []
        for i in v:
            if i.isdigit():
                # print(i) # DEBUG
                CHAIN += str(i) + ': ' + ruling[int(i)]
            else:
                i = int(i.split(",")[0])
                CHAIN += str(i) + ': ' + ruling[i]
            CHAIN += '\n'
            new_v.append(str(i))
        chain_dict.update({k: new_v})
        # each PROMPT contains one standalone textual CHAIN
        prompt_dict[k] = PRE + CHAIN
    input_dict = {'article_id': sample['article_id'],
                  'claim': sample['statement'],
                  'label': sample['annotated_label'],
                  'passage': PASSAGE,
                  'chain_dict': chain_dict,
                  'prompt_dict': prompt_dict}
    return input_dict


if __name__ == '__main__':
    merged_dataset = [format_one_ins(dataset.iloc[i]) for i in range(len(dataset))]

    # save merged dataset to file
    processed_path = 'data/TransExp_data'
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)

    file_path = os.path.join(processed_path, 'merged_data.pickle')
    with open(file_path, 'wb') as file:
        pickle.dump(merged_dataset, file)
