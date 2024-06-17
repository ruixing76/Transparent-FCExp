import re
import json
import numpy as np
from copy import deepcopy
from utils import *

# random seed
np.random.seed(2023)

# this path is only used to sample keys
raw_data_path = '../data/TransExp_data/raw_data/raw_dataset_uclaim_invalid_chain.json'


def remove_citation(sen_with_cit: str, tgt_cit: int):
    """
    only remove the target citation from the sentence
    :param sen_with_cit: a sentence with citation
    :param tgt_cit: the target citation to be removed
    """
    pattern = re.compile(rf'\s*\[{tgt_cit}\]\s*')
    # Replace the target citation with an empty string
    result = pattern.sub('', sen_with_cit)
    return result


def sample_normal_key(sample_size=24):
    # claims in raw_dataset_uclaim_invalid_chain.json are unique across both settings
    dataset = read_json(raw_data_path)
    # sample 10 keys from dataset
    sample_keys = np.random.choice(list(dataset.keys()), sample_size, replace=False)
    return sample_keys


def sample_control_key(previous_keys, sample_size=60):
    with (open(raw_data_path, 'r')) as f:
        dataset = json.load(f)
    # select instances that has only one citation
    one_cit_keys = []
    for key, samples in dataset.items():
        if len(samples['chain'][1]) == 1:
            one_cit_keys.append(key)
    # control data should not overlap with normal data
    for key in previous_keys:
        if key in one_cit_keys:
            one_cit_keys.remove(key)
    one_cit_keys = np.random.choice(one_cit_keys, sample_size, replace=False)
    return one_cit_keys


def create_sub_data(model_name, keys, target_filename, mode='core', data_type="normal"):
    if mode == 'core' or mode == 'full':
        filename = f'../data/TransExp_data/{model_name}_{mode}_processed_data.json'
    else:
        raise TypeError("mode should be one of core, full")
    dataset = read_json(filename)
    data = {k: dataset[k] for k in keys}
    # create 'masked_explanation' in the data, record masked_cit citation in ans_sens sentences
    no_cit = []
    for key, sample in data.items():
        try:
            masked_cit = sample['masked_cit'][model_name]
            ans_sens = sample['ans_sens'][model_name]
        except KeyError:
            # this sample didn't contain any citation
            # we will remove this sample from the data
            no_cit.append(key)
            # print(f"Sample {key} doesn't contain any citation.")
            continue
        sample['masked_explanation'] = {}
        if masked_cit == -1:
            sample['masked_explanation'][model_name] = sample['explanation'][model_name]
        else:
            masked_explanation = deepcopy(sample['explanation'][model_name])
            for sen_id in ans_sens:
                masked_explanation[sen_id] = remove_citation(masked_explanation[sen_id], masked_cit)
            sample['masked_explanation'][model_name] = masked_explanation
        # if it is a negative control dataset, remove sentences in explanation and masked_explanation
        # their ids are in the ans_sens
        if data_type == "normal":
            sample["type"] = "normal"
        elif data_type == "positive":
            sample["type"] = "positive"
        elif data_type == "negative":
            sample["type"] = "negative"
            for sen_id in ans_sens:
                sample['explanation'][model_name][sen_id] = ''
                sample['masked_explanation'][model_name][sen_id] = ''
        else:
            raise TypeError("data_type should be one of normal, positive, negative")

    # remove samples that don't contain any citation
    if len(no_cit) != 0:
        for key in no_cit:
            data.pop(key)
            print(f"Sample {key} is removed.")
        print(f"Total {len(no_cit)} samples are removed from the data because they don't contain any citation")
    write_json(data, f'./data/{target_filename}')


def main():
    model_name = 'llama2-70b'
    mode = 'full'
    normal_random = False  # whether normal keys are randomly sampled
    if normal_random:
        normal_keys = sample_normal_key(sample_size=100)
    else:
        normal_keys = read_json('hit_sample_keys.json')
    positive_keys = sample_control_key(normal_keys, sample_size=25)
    negative_keys = sample_control_key(np.concatenate((normal_keys, positive_keys)), sample_size=25)

    create_sub_data(model_name, normal_keys, f'{model_name}_{mode}_nor_data.json', mode=mode, data_type="normal")
    create_sub_data(model_name, positive_keys, f'{model_name}_{mode}_pos_data.json', mode=mode, data_type="positive")
    create_sub_data(model_name, negative_keys, f'{model_name}_{mode}_neg_data.json', mode=mode, data_type="negative")


if __name__ == '__main__':
    main()
