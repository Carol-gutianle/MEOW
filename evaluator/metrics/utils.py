import jsonlines

def read_jsonlines(data_path):
    data = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def read_lst(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        content = file.readlines()
    for line in content:
        data.append(line.strip())
    return data