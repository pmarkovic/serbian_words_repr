import json
import argparse
import linecache


def arg_parser():
    """ takes user input """

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_path", default="src/data/analogy_test_set.csv",
                        help="Path to the file where analogy test examples are stored (default=src/data/analogy_test.csv).")
    parser.add_argument("--results_path", default="results/sg_test2.json",
                        help="Path to the json file where the analogy test results are stored (default=results/sg_test2.json).")
    parser.add_argument("--example", default=0,
                        help="An example number for inspection (default=0).")

    args = parser.parse_args()

    return args


def check_example(args):
    example = linecache.getline(args.test_path, int(args.example)+1)
    example = example.strip().split(',')

    with open(args.results_path, 'r') as json_file:
        results = json.load(json_file)

    key = f"example_{args.example}"

    print(f"Category: {example[0]}")
    print(f"Example: {example[1]} is to {example[2]}, as {example[3]} is to ?")
    print(f"Expected output: {example[4]}")
    for pos, value in enumerate(results[key]):
        print(f"{pos+1}. {value}")


if __name__ == "__main__":
    """
    Script for checking retrieved most similar words for a particular example.
    """
    args = arg_parser()

    check_example(args)
