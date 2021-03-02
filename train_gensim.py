from gensim import models


class ReadCorpus:

  def __iter__(self):
    for line in open('/content/gdrive/MyDrive/Colab_Notebooks/train_set.txt'):
        yield line.strip().split()


if __name__ == "__main__":
    # TODO add argparse to take params from command line
    
    corpus_reader = ReadCorpus()

    gen_model = models.Word2Vec(sentences=corpus_reader, size=300, window=4, min_count=0, sg=1)

    gen_model.save("/content/gdrive/MyDrive/Colab_Notebooks/gen_sg300.txt")