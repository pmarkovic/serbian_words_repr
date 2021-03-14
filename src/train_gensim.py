from gensim.models import Word2Vec


class ReadCorpus:

  def __iter__(self):
    for line in open('/content/gdrive/MyDrive/Colab_Notebooks/train_set.txt'):
        yield line.strip().split()


if __name__ == "__main__":
    # TODO add argparse to take params from command line
    
    corpus_reader = ReadCorpus()

    gen_model = Word2Vec(sentences=corpus_reader, size=300, window=5, min_count=0)

    gen_model.save("/content/gdrive/MyDrive/Colab_Notebooks/gen_sg300.txt")