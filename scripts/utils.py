def counter_dict2list(counter_dict):
    result = []
    for k, v in counter_dict.items():
        result.extend([k] * int(v))

    return result