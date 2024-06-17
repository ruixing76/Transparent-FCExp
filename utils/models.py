import time
import logging
import openai
import tiktoken
from openai import OpenAI
from annotation.annotation_config import *

client = OpenAI(
    organization=API_ORG,
    api_key=API_KEY,
    max_retries=3
)

# model name
MODEL_LIST = ['gpt35', 'gpt4', 'llama2-70b']
MODES = ['core', 'full']
DATA_TYPE = ['nor', 'pos', 'neg']


def openai_generate(user_input, model='gpt-4o',
                    system_role="You are a helpful assistant.",
                    is_output_json=False):
    response_format = {"type": "json_object"} if is_output_json else {"type": "text"}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_input},
        ],
        response_format=response_format,
        max_tokens=1000,
        temperature=1
    )
    output_text = response.choices[0].message.content
    return output_text


def openai_generate_manual_retry(user_input, model='gpt-4o',
                                 system_role="You are a helpful assistant.",
                                 is_output_json=False,
                                 tries=3, wait_time=1):
    output_text = ""
    response_format = {"type": "json_object"} if is_output_json else {"type": "text"}
    for n in range(tries + 1):
        if n == tries:
            raise openai.APIError(f"Tried {tries} times.")
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_input},
                ],
                response_format=response_format,
                max_tokens=1000,
                temperature=1
            )
            output_text = completion.choices[0].message.content
        # please see all errors in openai __init__.py
        except (openai.APIError, openai.APIConnectionError,
                openai.RateLimitError, openai.Timeout) as e:
            logging.warning(e)
            logging.warning(f"Retry after {wait_time}s. (Trail: {n + 1})")
            time.sleep(wait_time)
            continue
        break
    return output_text


# utility functions
def update_prompt(prompt, info_dict):
    for item in info_dict.items():
        prompt = prompt.replace(f'<{item[0]}>', item[1])
    return prompt


def count_tokens(text, model='gpt-4'):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def calculate_cost(input_len, output_len, model='gpt-4'):
    """
    calculate cost of openai
    """
    # use gpt-4 model for now
    if model == 'gpt-4o':
        input_price = 5.0
        output_price = 15.0
    elif model == 'gpt-4-turbo':
        input_price = 10.0
        output_price = 30.0
    else:
        raise AssertionError("model not supported")
    cost = input_len / 1000000 * input_price + output_len / 1000000 * output_price
    return cost


if __name__ == '__main__':
    pass
