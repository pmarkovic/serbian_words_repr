import os
import time
import json
import logging
from lxml import etree
import concurrent.futures
from collections import defaultdict


fence = '=' * 20
CORPORA = ["srWaC1.1.01.xml", "srWaC1.1.02.xml", "srWaC1.1.03.xml", 
            "srWaC1.1.04.xml", "srWaC1.1.05.xml", "srWaC1.1.06.xml"]
CORPUS_PATH = os.path.join(os.path.dirname(os.getcwd()), "corpus")
DATA_PATH = os.path.join(os.getcwd(), "data")
MIN_LENGTH = 6
TOKENS_DIST_FILE = "_token_dist.json"
NUM_TOKEN = "NUM"


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", 
                    filename="preprocessing.log",
                    filemode="w",
                    level=logging.INFO, 
                    datefmt="%d-%b-%y %H:%M:%S")

def parse(iterator, dist):
    sent = list()

    for i in iterator:
        for line in i.split('\n'):
            tokens = line.split('\t')

            # Discard </g> tags
            if len(tokens) < 2:
                continue

            token = tokens[1]

            # Include only numeric or alphabetic tokens
            # Map all numeric tokens to the same value
            if token.isnumeric():
                token = NUM_TOKEN
            elif not token.isalpha():
                continue

            sent.append(token)
            dist[token] += 1

    return ' '.join(sent)


def parse_corpus(corpus_name):
    print(f"Processing {corpus_name}...")
    corpus_start_time = time.time()

    context = etree.iterparse(os.path.join(CORPUS_PATH, corpus_name), events=("start", "end"))
    corpus_id = corpus_name[:-4]
    data_dir = os.path.join(DATA_PATH, corpus_id)
    tokens_dist = defaultdict(float)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    with open(f"{os.path.join(data_dir, corpus_id)}.txt", 'a') as txt_file:
        for event, element in context:
            if event == "start" and element.tag == 's':
                sent = parse(element.itertext(), tokens_dist)
                txt_file.write(f"{sent}\n")
            elif event == "end":
                element.clear()
    
    with open(f"{os.path.join(data_dir, corpus_id)}{TOKENS_DIST_FILE}", 'w') as json_file:
        json.dump(tokens_dist, json_file, indent=4)
    
    print(f"Corpus {corpus_id} total time: {round(time.time() - corpus_start_time, 2)}s")


def transform_corpora():
    start_transform_time = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(parse_corpus, CORPORA)

    logging.info(f"Total transformation time: {round(time.time() - start_transform_time, 2)}s")


def merge_tokens_dist():
    print("Start merging tokens distributions...")
    merging_start_time = time.time()
    total_tokens_dist = defaultdict(float)

    for dir in os.listdir(DATA_PATH):
        dir_path = os.path.join(DATA_PATH, dir)

        if not os.path.isdir(dir_path):
            continue
        
        for file in os.listdir(dir_path):
            if file.endswith(TOKENS_DIST_FILE):
                with open(f"{os.path.join(dir_path, file)}", 'r') as json_file:
                    tokens_dist = json.load(json_file)

                    for key, value in tokens_dist.items():
                        total_tokens_dist[key] += value

    with open(f"{os.path.join(DATA_PATH, 'tokens_dist.json')}", 'w') as json_file:
        json.dump(total_tokens_dist, json_file, indent=4)

    logging.info(f"Total merging time: {round(time.time() - merging_start_time, 2)}s")
    print("Finished merging tokens distributions!")


def create_train_sets():
    pass


if __name__ == "__main__":
    logging.info(f"{fence}Program started{fence}")
    program_start_time = time.time()

    transform_corpora()
    merge_tokens_dist()

    logging.info(f"Total time: {round(time.time() - program_start_time, 2)}s")
    logging.info(f"{fence}Program finished{fence}")
