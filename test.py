import json
if __name__ == '__main__':
    path = 'data/Bart-FiD/test.json'
    with open(path, 'r') as f:
        res = json.load(f)
