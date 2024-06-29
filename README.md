# Transparent-FCExp

![License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)


This repo contains code and data for the paper [Evaluating Transparency of Machine Generated Fact Checking Explanations](https://arxiv.org/abs/2406.12645).

## Abstract
An important factor when it comes to generating fact-checking explanations is the selection of evidence: intuitively, high-quality explanations can only be generated given the right evidence. In this work, we investigate the impact of human-curated vs. machine-selected evidence for explanation generation using large language models. To assess the quality of explanations, we focus on transparency (whether an explanation cites sources properly) and utility (whether an explanation is helpful in clarifying a claim). Surprisingly, we found that large language models generate similar or higher quality explanations using machine-selected evidence, suggesting carefully curated evidence (by humans) may not be necessary. That said, even with the best model, the generated explanations are not always faithful to the sources, suggesting further room for improvement in explanation generation for fact-checking.
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
@misc{xing2024evaluating,
      title={Evaluating Transparency of Machine Generated Fact Checking Explanations}, 
      author={Rui Xing and Timothy Baldwin and Jey Han Lau},
      year={2024},
      eprint={2406.12645},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```
