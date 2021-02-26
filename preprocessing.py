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
TOTAL_TOKENS = 690000
ALPHA = 3/4


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

    tokens_dist = {key: tokens_dist[key] for key in sorted(total_tokens_dist, key=total_tokens_dist.__getitem__, reverse=True)[:TOTAL_TOKENS]}

    with open(f"{os.path.join(DATA_PATH, 'tokens_dist.json')}", 'w') as json_file:
        json.dump(total_tokens_dist, json_file, indent=4)

    logging.info(f"Total merging time: {round(time.time() - merging_start_time, 2)}s")
    print("Finished merging tokens distributions!")


def clean_corpus():
    print("Starting to clean the corpus...")
    create_start_time = time.time()
    total_num_sents = 0.0
    avg_sent_len = 0.0
    total_num_tokens = 0.0
    train_num_sents = 0.0
    avg_train_sent_len = 0.0
    total_num_train_tokens = 0.0
    tokens_dist = {}
    noise_dist = defaultdict(float)

    with open(f"{os.path.join(DATA_PATH, 'tokens_dist.json')}", 'r') as json_file:
        tokens_dist = json.load(json_file)

    for dir in os.listdir(DATA_PATH):
        dir_path = os.path.join(DATA_PATH, dir)

        if not os.path.isdir(dir_path):
            continue

        print(f"Processing {dir} directory...")
        for file in os.listdir(dir_path):
            if not file.endswith(".txt"):
                continue
            
            with open(f"{os.path.join(dir_path, file)}", 'r') as txt_file:
                for line in txt_file:
                    tokens = line.split(' ')
                    total_num_sents += 1
                    total_num_tokens += len(tokens)
                    avg_sent_len += len(tokens)
                    train_sent = []

                    for token in tokens:
                        if token in tokens_dist:
                            train_sent.append(token)
                    
                    if len(train_sent) > 5:
                        train_num_sents += 1
                        avg_train_sent_len += len(train_sent)
                        total_num_train_tokens += len(train_sent)

                        for token in train_sent:
                            noise_dist[token] += 1
                    
                        with open(f"{os.path.join(DATA_PATH, 'train_set.txt')}", 'a') as txt_file:
                            txt_file.write(f"{' '.join(train_sent)}\n")
    
    print("Creating noise distribution...")
    # Create noise distribution
    Z = 0.0
    for key in noise_dist.keys():
        noise_dist[key] /= total_num_train_tokens
        noise_dist[key] **= ALPHA
        Z += noise_dist[key]
  
    for key in noise_dist.keys():
        noise_dist[key] /= Z

    with open(f"{os.path.join(DATA_PATH, 'noise_dist.json')}", 'w') as json_file:
        json.dump(noise_dist, json_file, indent=4)

    print(f"Total number of sentences: {total_num_sents}")
    print(f"Average sentence length: {round(avg_sent_len / total_num_sents, 2)}") 
    print(f"Total number of tokens: {total_num_tokens}")
    print()
    print(f"Total number of train sentences: {train_num_sents}")
    print(f"Average train sentence length: {round(avg_train_sent_len / train_num_sents, 2)}")
    print()

    logging.info(f"Total cleaning time: {round(time.time() - create_start_time, 2)}s")
    print("Finished cleaning the corpus!")


if __name__ == "__main__":
    logging.info(f"{fence}Program started{fence}")
    program_start_time = time.time()

    #transform_corpora()
    #merge_tokens_dist()
    clean_corpus()

    logging.info(f"Total time: {round(time.time() - program_start_time, 2)}s")
    logging.info(f"{fence}Program finished{fence}")
