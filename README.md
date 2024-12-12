# Transparent-FCExp

![License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)


This repo contains code and data for the paper [Evaluating Evidence Attribution in Generated Fact Checking Explanations](https://arxiv.org/abs/2406.12645).

## Abstract
Automated fact-checking systems often struggle with trustworthiness, as their generated explanations can include hallucinations. In this work, we explore evidence attribution for fact-checking explanation generation. We introduce a novel evaluation protocol, citation masking and recovery, to assess attribution quality in generated explanations. We implement our protocol using both human annotators and automatic annotators, and find that LLM annotation correlates with human annotation, suggesting that attribution assessment can be automated. Finally, our experiments reveal that: (1) the best-performing LLMs still generate explanations with inaccurate attributions; and (2) human-curated evidence is essential for generating better explanations.
## Installation
```
git clone git@github.com:ruixing76/Transparent-FCExp.git && cd Transparent-FCExp && pip install -r requirement.txt
```

## Data
### Original data
The original PolitiHop data can be downloaded here: https://github.com/copenlu/politihop. Please put the data under `./data/PolitiHop_data/`.

### Generated Explanation
The generated explanation data is mainly used in our work, which is stored here: `./data/TransExp_data/{model_name}_{setting}_data.json`.

- `model_name` should be `gpt4`, `gpt35` or `llama2-70b`.
- `setting` should be `core` (Human setting) or `full` (Machine setting).

### Data Format
```json
"CLAIM_ID": {
        "claim": "claim content",
        "label": "claim veracity label from {true, false and half-true}",
        // No. 12 reason cited in explanation is masked
        "masked_reason": 12,
        // answer index in "explanation"
        "ans_sens": [
            1 
        ],
        "core_reasons": [
            "12: No.12 core reason content"
        ],
        // "1: explanation sentence [12]" is the ground-truth
        "explanation": [
            "0: explanation sentence",
            "1: explanation sentence [12]",
            "2: explanation sentence"
        ],
        // top 2 annotator's choices, -2 indicates 'no citation'
        "top_choice": [
            -2
            2
        ]
    }
```

## Preprocessing
```
python preprocess_politihop.py
```
We recommend to use our preprocessed data under: `./data/TransExp_data/raw_dataset/raw_dataset.json`.

## Explanation Generation
Generate explanations using:
```
python generate_explanation.py -model_name llama2-70b -output_dir output_dir
```
- `-model_name` should be `gpt4`, `gpt35` or `llama2-70b`.
- `-output_dir` is output directory for generated explanation.

Postprocess generated explanation, extract, mask and sample citation.
```
python postprocess_generation.py -model_name llama2-70b -output_dir output_dir
```
- `-model_name` should be `gpt4`, `gpt35` or `llama2-70b`.
- `-output_dir` is output directory for postprocessed explanation.

## Annotation
Generate annotation data using:
```
cd ./annotation
python create_annotation_data.py
```

Annotation is performed on [Amazon Mechanical Turk](https://www.mturk.com/). The webpage template can be found under `./annotation/annotation_platform.html`. Generate annotation webpage using:
```
python create_HIT.py
```

## Cite
If you find this work useful, please kindly cite our paper.
```
@misc{xing2024evaluatingevidenceattributiongenerated,
      title={Evaluating Evidence Attribution in Generated Fact Checking Explanations}, 
      author={Rui Xing and Timothy Baldwin and Jey Han Lau},
      year={2024},
      eprint={2406.12645},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.12645}, 
}
```
