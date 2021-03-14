import argparse
from gensim.models import Word2Vec


def arg_parser():
    """ takes user input """

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="data/train_set.txt",
                        help="Path to the training data (default=data/train_set.txt).")
    parser.add_argument("--weights_path", default="../weights/gen_sg300.txt",
                        help="Path to the word2ind dict (default=../weights/gen_sg300.txt).")
    parser.add_argument("--dim", default=300,
                        help="Embeddings dimension (default=300).")
    parser.add_argument("--ws", default=3,
                        help="Maximal window size (default=3).")
    parser.add_argument("--is_sg", default=False, action="store_true",
                        help="Flag to indicate if skipgram model will be used (default=False).")

    args = parser.parse_args()

    return args


class ReadCorpus:
  """
  Class for reading train data.
  """

  def __init__(self, path):
      self.path = path

  def __iter__(self):
    for line in open(self.path):
        yield line.strip().split()


if __name__ == "__main__":
    """
    Script for training gensim models.
    """

    args = arg_parser()
    
    corpus_reader = ReadCorpus(args.train_path)

    gen_model = Word2Vec(sentences=corpus_reader, 
                         size=args.dim, 
                         window=args.ws, 
                         min_count=0, 
                         sg=args.is_sg)

    gen_model.save(args.weights_path)