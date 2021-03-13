import os
import time
import json
import argparse
import concurrent.futures

import numpy as np


def arg_parser():
    """ takes user input """

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", default="./data",
                        help="Path to the training data (default=./data).")
    parser.add_argument("--name", default="train_set.txt",
                        help="Name of training set (default=train_set.txt)")
    parser.add_argument("--sep_file", default=False, action="store_true",
                        help="Flag to indicate if separated files of train set should be created (default=False).")
    parser.add_argument("--is_sg", default=False, action="store_true",
                        help="Flag to indicate which model to use (default=False).")

    args = parser.parse_args()

    return args


def create_sep_files(args):
    train_set_path = os.path.join(args.path, args.name)
    
    with open(train_set_path, 'r') as txt_file:
        train_set = txt_file.readlines()

    train_set_size = len(train_set) - 1
    batch = train_set_size // 10

    for i in range(0, train_set_size, batch):
        file_name = f"file_{i // batch}.txt"

        with open(os.path.join(args.path, file_name), 'a') as txt_file:
            for sent in train_set[i:min(train_set_size, i + batch)]:
                txt_file.write(f"{sent}")


def init_dictionaries(word2ind_path, noise_dist_path):
    with open(word2ind_path, 'r') as json_file:
        word2ind = json.load(json_file)

    with open(noise_dist_path, 'r') as json_file:
        noise_dist = json.load(json_file)

    return word2ind, noise_dist


def cbow_examples(file_path):
    start_time = time.time()

    file_name = f"cbow_{file_path[-5]}.csv"
    cbow_file_path = os.path.join("./data", file_name)
    max_ws = 3

    print(f"Start processing for {file_name}...")

    # Need for sampling negative examples
    word2ind, noise_dist = init_dictionaries("./data/word2ind.json", "./data/noise_dist.json")
    tokens = list(noise_dist.keys())
    tokens_prob = list(noise_dist.values())

    with open(cbow_file_path, 'a') as csv_file:
        with open(file_path, 'r') as txt_file:
            for line in txt_file:
                words = line.strip().split(' ')

                for pos, word in enumerate(words):
                        center_ind = word2ind[word]
                        context_words = []

                        # Choose window size dynamically
                        ws = np.random.randint(2, min(max_ws+1, len(words)))

                        # Look for words around the center word
                        for w in range(-ws, ws+1):
                            context_pos = pos + w 

                            # Checks for index not to go out of bound 
                            # and to be different from center word's index
                            if 0 <= context_pos < len(words) and context_pos != pos:
                                context_words.append(str(word2ind[words[context_pos]]))

                        samples_ind = [str(word2ind[sample]) 
                                        for sample in np.random.choice(tokens, size=5, p=tokens_prob)]
                        
                        output_line = f"{center_ind};{','.join(context_words)};{','.join(samples_ind)}\n"
                        csv_file.write(output_line)
    
    print(f"Finished processing {file_name}! Time: {round(time.time() - start_time, 2)}s")


def sg_examples(file_path):
    start_time = time.time()

    file_name = f"sg_{file_path[-5]}.csv"
    sg_file_path = os.path.join("./data", file_name)
    max_ws = 3

    print(f"Start processing for {file_name}...")

    # Need for sampling negative examples
    word2ind, noise_dist = init_dictionaries("./data/word2ind.json", "./data/noise_dist.json")
    tokens = list(noise_dist.keys())
    tokens_prob = list(noise_dist.values())

    with open(sg_file_path, 'a') as csv_file:
        with open(file_path, 'r') as txt_file:
            for line in txt_file:
                words = line.strip().split(' ')

                for pos, word in enumerate(words):
                    center_ind = word2ind[word]

                    # Choose window size dynamically
                    ws = np.random.randint(2, min(max_ws+1, len(words)))

                    samples_ind = [str(word2ind[sample]) 
                                    for sample in np.random.choice(tokens, size=5, p=tokens_prob)]

                    # Look for words around the center word
                    for w in range(-ws, ws+1):
                        context_pos = pos + w 

                        # Checks for index not to go out of bound 
                        # and to be different from center word's index
                        if 0 <= context_pos < len(words) and context_pos != pos:
                            context_ind = word2ind[words[context_pos]]
                    
                            output_line = f"{center_ind};{context_ind};{','.join(samples_ind)}\n"
                            csv_file.write(output_line)
    
    print(f"Finished processing {file_name}! Time: {round(time.time() - start_time, 2)}s")


def crete_examples(args):
    start_time = time.time()

    print("Start creating examples...")
    print(f"Is sg: {args.is_sg}")

    files = [os.path.join(args.path, f"file_{i}.txt") for i in range(10)]
    model_examples = sg_examples if args.is_sg else cbow_examples

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(model_examples, files)

    print(f"Finished creating examples. Total time: {round(time.time() - start_time, 2)}s")


if __name__ == "__main__":
    args = arg_parser()

    if args.sep_file:
        create_sep_files(args)
    
    crete_examples(args)
