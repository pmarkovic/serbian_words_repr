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
DATA_PATH = os.path.join(os.getcwd(), "data")
TOKENS_DIST_FILE = "_token_dist.json"

# Constants
# For logging purpose
FENCE = '=' * 20
# Token value for all number values in the corpus
NUM_TOKEN = "NUM"
# Limit number of tokens in vocabulary 
# to include only tokens that appear more than 5 times
LIMIT = 100000
# Used for noise distribution
ALPHA = 3/4
# Minimum length of sentences 
# that should be included into training examples
MIN_LENGTH = 6


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
    Transform_corpus function is the first step in the preprocessing phase.
    The corpus consists of 6 separate files in xml format and sentences are stored in <s> tags.
    This function simply creates separate processes to parse files.
    """
    
    start_transform_time = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(parse_corpus, CORPORA)

    logging.info(f"Total transformation time: {round(time.time() - start_transform_time, 2)}s")


def merge_tokens_dist():
    """
    Merge_tokens_dist is the second step in the preprocessing phase.
    Since tokens distributions of files are created separately, this function
    merge them to the tokens distribution of the whole corpus. And stores it
    as a json file in the data directory.
    """

    print("Start merging tokens distributions...")
    merging_start_time = time.time()
    total_tokens_dist = defaultdict(float)

    # List directories in the data directory
    for dir in os.listdir(DATA_PATH):
        dir_path = os.path.join(DATA_PATH, dir)

        # Ignore any file
        if not os.path.isdir(dir_path):
            continue
        
        for file in os.listdir(dir_path):
            # Check for tokens distribution file
            if file.endswith(TOKENS_DIST_FILE):
                with open(f"{os.path.join(dir_path, file)}", 'r') as json_file:
                    tokens_dist = json.load(json_file)

                    # Merge tokens into single dictionary
                    for key, value in tokens_dist.items():
                        total_tokens_dist[key] += value

    # Use only 100000 unique tokens for training (tokens that appear more than 5 times)
    total_tokens_dist = {key: total_tokens_dist[key] \
                  for key in sorted(total_tokens_dist, key=total_tokens_dist.__getitem__, reverse=True)[:LIMIT]}

    # Store the tokens distribution of the whole corpus
    with open(f"{os.path.join(DATA_PATH, 'tokens_dist.json')}", 'w') as json_file:
        json.dump(total_tokens_dist, json_file, indent=4)

    logging.info(f"Total merging time: {round(time.time() - merging_start_time, 2)}s")
    print("Finished merging tokens distributions!")


def clean_corpus():
    """
    Clean_corpus is the third step in the preprocessing phase.
    The function merge sentences from all processed files and 
    discard tokens from sentences if they are not included in tokens distribution.
    The first 100000 sentences from every file which length is at least 6 tokens are included.
    Lastly, the function prints minimum stats regarding corpus.
    """

    print("Starting to clean the corpus...")
    clean_start_time = time.time()
    
    # Basic stats of the corpus
    total_num_sents = 0.0
    avg_sent_len = 0.0
    total_num_tokens = 0.0
    train_num_sents = 0.0
    avg_train_sent_len = 0.0
    tokens_dist = {}

    # Required for noise distribution
    total_num_train_tokens = 0.0
    noise_dist = defaultdict(float)

    # Load tokens distribution
    with open(f"{os.path.join(DATA_PATH, 'tokens_dist.json')}", 'r') as json_file:
        tokens_dist = json.load(json_file)

    for dir in os.listdir(DATA_PATH):
        dir_path = os.path.join(DATA_PATH, dir)

        # Ignore any file
        if not os.path.isdir(dir_path):
            continue

        print(f"Processing {dir} directory...")
        for file in os.listdir(dir_path):

            # Ignore non-text files
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
                        # Discard tokens which are not in the final distribution
                        if token in tokens_dist:
                            train_sent.append(token)
                    
                    # Discard sentences with legth less than 6
                    if len(train_sent) < MIN_LENGTH:
                        continue

                    train_num_sents += 1
                    avg_train_sent_len += len(train_sent)
                    total_num_train_tokens += len(train_sent)

                    for token in train_sent:
                        noise_dist[token] += 1
                
                    with open(f"{os.path.join(DATA_PATH, 'train100.txt')}", 'a') as txt_file:
                        txt_file.write(f"{' '.join(train_sent)}\n")
                    
                    # Include only first 100000 from every file
                    if train_num_sents % LIMIT == 0:
                        return
    
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

    logging.info(f"Total cleaning time: {round(time.time() - clean_start_time, 2)}s")
    print("Finished cleaning the corpus!")


def make_vocabulary():
    word2ind = dict()
    counter = 0
    data_path = os.path.join(DATA_PATH, "train_set.txt")

    with open(data_path, 'r') as file:
        for sent in file:
            for word in sent.strip().split(' '):
                if word not in word2ind:
                    word2ind[word] = counter
                    counter += 1

    with open("data/word2ind.json", 'w', encoding="utf-8") as json_file:
        json.dump(word2ind, json_file, indent=4)


if __name__ == "__main__":
    logging.info(f"{FENCE}Program started{FENCE}")
    program_start_time = time.time()

    #transform_corpus()
    #merge_tokens_dist()
    clean_corpus()
    #make_vocabulary()

    logging.info(f"Total time: {round(time.time() - program_start_time, 2)}s")
    logging.info(f"{FENCE}Program finished{FENCE}")
