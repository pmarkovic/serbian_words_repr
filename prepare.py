import os
import json
import time
import argparse
import numpy as np


NOISE_DIST_PATH = os.path.join(os.getcwd(), "data", "noise_dist.json")


def arg_parser():
    """ takes user input """

    parser = argparse.ArgumentParser()

    parser.add_argument("--examples_path", default="data/examples.json", 
                        help="Path to the json file where to store training examples (default=data/examples.json).")
    parser.add_argument("--data_path", default="data/train_set.txt",
                        help="Path to the file where training data are stored (default=data/train_set.txt).")
    parser.add_argument("--seed", default=892,
                        help="Seed for random numbers generator (default=892).")
    parser.add_argument("--wind_size", default=5, 
                        help="Max window size for surrounding context words (default=5).")
    parser.add_argument("--neg_sample", default=5,
                        help="Number of negative samples to pick (default=5).")

    args = parser.parse_args()

    return args


def make_vocabulary(args):
    word2ind = dict()
    counter = 0
    data_path = args.data_path

    with open(data_path, 'r') as file:
        for sent in file:
            for word in sent.strip().split(' '):
                if word not in word2ind:
                    word2ind[word] = counter
                    counter += 1

    return word2ind


def create_train_examples(args):
    np.random.seed(args.seed)
    window_size = args.wind_size
    data_path = args.data_path
    neg_sample = args.neg_sample
    train_data = {}
    examples_counter = 0

    with open(NOISE_DIST_PATH, 'r') as json_file:
        noise_dist = json.load(json_file)
    tokens = list(noise_dist.keys())
    tokens_prob = list(noise_dist.values())

    word2ind = make_vocabulary(args)

    with open(data_path, 'r') as file:
        for i, sent in enumerate(file):
            if i % 1000000 == 0:
                print(f"Processed {i} lines...")

            words = sent.strip().split(' ')

            for pos, word in enumerate(words):
                center_ind = word2ind[word]
                context_ind = []
                ws = np.random.randint(2, window_size+1)

                for w in range(-ws, ws+1):
                    context_pos = pos + w 

                    if 0 <= context_pos < len(words) and context_pos != pos:
                        context_ind.append(word2ind[words[context_pos]])

                samples_ind = [word2ind[sample] \
                                  for sample in np.random.choice(tokens, size=neg_sample, p=tokens_prob)]
                train_data[f"example{examples_counter}"] = {"center": center_ind, 
                                                            "context": context_ind, 
                                                            "neg_samples": samples_ind}
                examples_counter += 1

    with open(args.examples_path, 'w') as json_file:
        json.dump(train_data, json_file, indent=4)


if __name__ == "__main__":
    start_time = time.time()
    args = arg_parser()

    create_train_examples(args)

    print(f"Total time: {round(time.time() - start_time, 2)}s")
