import csv
import json
import argparse

from gensim.models.word2vec import Word2Vec

import torch
import torch.nn.functional as F



def arg_parser():
    """ takes user input """

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights_path", default="weights/my_sg300.txt",
                        help="Path to the training data (default=weights/my_sg300.txt).")
    parser.add_argument("--w2i_path", default="src/data/word2ind.json",
                        help="Path to the word2ind dict (default=data/word2ind.json).")
    parser.add_argument("--i2w_path", default="src/data/ind2word.json",
                        help="Path to the ind2word dict (default=data/ind2word.json).")
    parser.add_argument("--test_set_path", default="src/data/analogy_test_set.csv",
                        help="Path to the analogy test data (default=data/analogy_test_set.csv).")
    parser.add_argument("--results_path", default="src/results/analogy_test.json",
                        help="Path to the json file where to store the analogy test results (default=results/analogy_test.json)")
    parser.add_argument("--is_gen", default=False, action="store_true",
                        help="Flag to indicate if gensim weights will be used (default=False).")

    args = parser.parse_args()

    return args


def analogy_test(args):
    # Load weights
    if args.is_gen:
        results_list, results_dict = use_gen_model(args)
    else:
        results_list, results_dict = use_weights(args)

    # Save results
    # Results per category are stored in rows
    results = torch.tensor(results_list).view(5, 5)
    total_correct = results.sum().numpy().tolist()
    correct_per_category = results.sum(dim=1).numpy().tolist()

    results_dict["result"] = {"total_correct": total_correct, 
                              "correct_per_category": correct_per_category}

    with open(args.results_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)


def use_weights(args):
    weights = torch.load(args.weights_path)

    # Load vocabularies
    with open(args.w2i_path, 'r') as json_file:
        word2ind = json.load(json_file)

    with open(args.i2w_path, 'r') as json_file:
        ind2word = json.load(json_file)

    # Load test data
    examples = []
    with open(args.test_set_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        for row in reader:
            examples.append([word2ind[word] for word in row[1:]])

    # Perform test
    results_list = []
    results_dict = {}
    for i, example in enumerate(examples):
        one_hot = F.one_hot(torch.tensor(example), num_classes=len(word2ind)).float()
        vectors = one_hot @ weights

        y = vectors[1] - vectors[0] + vectors[2]
        dist = F.cosine_similarity(weights, y.unsqueeze(dim=0))

        index_sorted = torch.argsort(dist, descending=True)
        top_10 = index_sorted[:10]
        top_10_words = [ind2word[str(int(ind))] for ind in top_10]

        results_list.append(int(top_10[0] == example[3]))
        results_dict[f"example_{i}"] = top_10_words

        print(f"Example: {' - '.join(ind2word[str(ind)] for ind in example)}")
        print(f"Most similar: {top_10_words}")

    return results_list, results_dict


def use_gen_model(args):
    gen_model = Word2Vec.load(args.weights_path)

    results_list = []
    results_dict = {}

    with open(args.test_set_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        for ind, row in enumerate(reader):
            print(f"Row: {' '.join(row[1:])}")

            most_sim = gen_model.most_similar(positive=[row[2], row[3]], negative=[row[1]])
            print(f"Most similar: {most_sim}")

            results_list.append(int(row[4] == most_sim[0][0]))
            results_dict[f"example_{ind}"] = [s[0] for s in most_sim]

    return results_list, results_dict


if __name__ == "__main__":
    """
    Script to perform analogy test.
    """
    
    args = arg_parser()

    analogy_test(args)

    
