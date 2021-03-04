import os
import time
import json
import logging
from lxml import etree
import concurrent.futures
from collections import defaultdict


# Paths and names
CORPORA = ["srWaC1.1.01", "srWaC1.1.02", "srWaC1.1.03", 
            "srWaC1.1.04", "srWaC1.1.05", "srWaC1.1.06"]
CORPUS_PATH = os.path.join(os.path.dirname(os.getcwd()), "corpus")
DATA_PATH = "../data/srWaC1.1.01"
TOKENS_DIST_FILE = "_token_dist.json"

# Constants
# For logging purpose
FENCE = '=' * 20
# Token value for all number values in the corpus
NUM_TOKEN = "NUM"


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", 
                    filename="preprocessing.log",
                    filemode="w",
                    level=logging.INFO, 
                    datefmt="%d-%b-%y %H:%M:%S")


def parse(iterator, tokens_dist):
    """
    Helper function to parse single <s> tag
    to extract a sentence from it.
    ...

    Parameters:
    -----------
    iterator: iter
        Iterator object of <s> tag to retrieve its elements.
    tokens_dist: dict
        A dictionary to store distribution of tokens.

    Return:
    -------
    str
        Actual sentence from the <s> tag.
    """
    
    sent = list()

    for i in iterator:
        for line in i.split('\n'):
            tokens = line.split('\t')

            # Discard </g> tags
            if len(tokens) < 2:
                continue

            # At position 1 is lowercased word
            token = tokens[1]

            # Include only numeric or alphabetic tokens
            # Map all numeric tokens to the same value
            if token.isnumeric():
                token = NUM_TOKEN
            elif not token.isalpha():
                continue

            sent.append(token)
            tokens_dist[token] += 1

    return ' '.join(sent)


def parse_corpus(file_name):
    """
    Helper function that parse a single corpus file,
    since the whole corpus is divided into 6 separate files.
    For every file, new directory will be created to store parsed text
    and a json file for tokens distribution of that file.
    ...

    Parameter:
    ----------
    file_name: str
        A name of a particular file of the corpus (without extension).
    """

    print(f"Processing {file_name}...")
    corpus_start_time = time.time()

    context = etree.iterparse(os.path.join(CORPUS_PATH, f"{file_name}.xml"), events=("start", "end"))
    data_dir = os.path.join(DATA_PATH, file_name)
    tokens_dist = defaultdict(float)

    # Create separate directory for every file
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    # Parse the file to extract sentences from <s> tags and
    # write to a new .txt file where every sentence will be in separate line
    with open(f"{os.path.join(data_dir, file_name)}.txt", 'a') as txt_file:
        for event, element in context:
            if event == "start" and element.tag == 's':
                sent = parse(element.itertext(), tokens_dist)
                txt_file.write(f"{sent}\n")
            elif event == "end":
                element.clear()
    
    # Save tokens distribution (counts) from the current file
    with open(f"{os.path.join(data_dir, file_name)}{TOKENS_DIST_FILE}", 'w') as json_file:
        json.dump(tokens_dist, json_file, indent=4)
    
    print(f"Corpus {file_name} total time: {round(time.time() - corpus_start_time, 2)}s")


def transform_corpus():
    """
    The corpus consists of 6 separate files in xml format and sentences are stored in <s> tags.
    This function simply creates separate processes to parse files.
    """
    
    start_transform_time = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(parse_corpus, CORPORA)

    logging.info(f"Total transformation time: {round(time.time() - start_transform_time, 2)}s")


def get_train_set(data_path, corpus_name, limit, min_len, max_len, alpha):
    """
    The function merge sentences from all processed files and 
    discard tokens from sentences if they are not included in tokens distribution.
    The first 100000 sentences from every file which length is at least 6 tokens are included.
    Lastly, the function prints minimum stats regarding corpus.
    """

    corpus_file = os.path.join(data_path, f"{corpus_name}.txt")
    token_dist_file = os.path.join(data_path, f"{corpus_name}_token_dist.json")
    train_set_file = os.path.join(data_path, "train_set.txt")
    noise_dist_file = os.path.join(data_path, 'noise_dist.json')

    num_train_sent = 0
    avg_train_sent_len = 0
    
    # Required for noise distribution
    total_num_train_tokens = 0.0
    noise_dist = defaultdict(float)

    with open(token_dist_file, 'r') as json_file:
        tokens_dist = json.load(json_file)

    # Use the specified number of the most frequent unique tokens for training
    tokens_dist = {key: tokens_dist[key] \
                  for key in sorted(tokens_dist, key=tokens_dist.__getitem__, reverse=True)[:limit]}

    with open(corpus_file, 'r') as txt_file:
        for line in txt_file:
            # Discard tokens which are not in the final distribution
            train_sent = [token for token in line.split(' ') if token in tokens_dist]

            if min_len < len(train_sent) <= max_len:
                num_train_sent += 1
                avg_train_sent_len += len(train_sent)
                total_num_train_tokens += len(train_sent)

                for token in train_sent:
                    noise_dist[token] += 1
            
                with open(train_set_file, 'a') as txt_file:
                    txt_file.write(f"{' '.join(train_sent)}\n")

            if num_train_sent == limit:
                break

    # Create noise distribution
    Z = 0.0
    for key in noise_dist.keys():
        noise_dist[key] /= total_num_train_tokens
        noise_dist[key] **= alpha
        Z += noise_dist[key]
  
    for key in noise_dist.keys():
        noise_dist[key] /= Z

    with open(noise_dist_file, 'w') as json_file:
        json.dump(noise_dist, json_file, indent=4)

    print(f"Total number of train sentences: {limit}")
    print(f"Average train sentence length: {round(avg_train_sent_len / limit, 2)}")
    print(f"Number of unique tokens in train set: {len(noise_dist.keys())}")


def make_vocabulary(data_path):
    train_set_file = os.path.join(data_path, "train_set.txt")
    word2ind_file = os.path.join(data_path, "word2ind.json")

    word2ind = dict()
    counter = 0

    with open(train_set_file, 'r') as file:
        for sent in file:
            for word in sent.strip().split(' '):
                if word not in word2ind:
                    word2ind[word] = counter
                    counter += 1

    with open(word2ind_file, 'w', encoding="utf-8") as json_file:
        json.dump(word2ind, json_file, indent=4)


if __name__ == "__main__":
    logging.info(f"{FENCE}Program started{FENCE}")
    program_start_time = time.time()

    #transform_corpus()
    get_train_set("./data", "srWaC1.1.01", 100000, 5, 20, 3/4)
    make_vocabulary("./data")

    logging.info(f"Total time: {round(time.time() - program_start_time, 2)}s")
    logging.info(f"{FENCE}Program finished{FENCE}")
