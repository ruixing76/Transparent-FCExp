import json


def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, file_path, is_friendly_format=True):
    if is_friendly_format:
        indent = 4
    else:
        indent = None
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)
    print(f"Data is saved to {file_path}")