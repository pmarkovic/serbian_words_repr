Running options: 
train_model.py [-h] [--data_path DATA_PATH]
                      [--weights_path WEIGHTS_PATH] [--is_sg]
                      [--embed_dim EMBED_DIM] [--epochs EPOCHS] [--lr LR]
                      [--seed SEED] [--wind_size WIND_SIZE]
                      [--batch_size BATCH_SIZE] [--sample SAMPLE]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the training data
                        (default=data/cbow_examples.csv).
  --weights_path WEIGHTS_PATH
                        Path where to store trained weights/embeddings
                        (default=weights/cbow300_).
  --is_sg               Flag to indicate which model to use (default=False).
  --embed_dim EMBED_DIM
                        Dimension of embeddings (default=300).
  --epochs EPOCHS       Number of training epochs (default=5).
  --lr LR               Learning rate for optimizer (default=0.025).
  --seed SEED           Seed for random numbers generator (default=892).
  --wind_size WIND_SIZE
                        Maximal window size for context (default=3).
  --batch_size BATCH_SIZE
                        Size of batches during training (default=128).
  --sample SAMPLE       Number of training examples per epoch
                        (default=1000000).


train_gensim.py [-h] [--train_path TRAIN_PATH]

                       [--weights_path WEIGHTS_PATH] [--dim DIM] [--ws WS]
                       [--is_sg]

optional arguments:
  -h, --help            show this help message and exit
  --train_path TRAIN_PATH
                        Path to the training data
                        (default=data/train_set.txt).
  --weights_path WEIGHTS_PATH
                        Path to the word2ind dict
                        (default=../weights/gen_sg300.txt).
  --dim DIM             Embeddings dimension (default=300).
  --ws WS               Maximal window size (default=3).
  --is_sg               Flag to indicate if skipgram model will be used
                        (default=False).


test.py [-h] [--weights_path WEIGHTS_PATH] [--w2i_path W2I_PATH]
               [--i2w_path I2W_PATH] [--test_set_path TEST_SET_PATH]
               [--results_path RESULTS_PATH] [--is_gen]

optional arguments:
  -h, --help            show this help message and exit
  --weights_path WEIGHTS_PATH
                        Path to the training data
                        (default=weights/my_sg300.txt).
  --w2i_path W2I_PATH   Path to the word2ind dict
                        (default=data/word2ind.json).
  --i2w_path I2W_PATH   Path to the ind2word dict
                        (default=data/ind2word.json).
  --test_set_path TEST_SET_PATH
                        Path to the analogy test data
                        (default=data/analogy_test_set.csv).
  --results_path RESULTS_PATH
                        Path to the json file where to store the analogy test
                        results (default=results/analogy_test.json)
  --is_gen              Flag to indicate if gensim weights will be used
                        (default=False).


create_examples.py [-h] [--path PATH] [--name NAME] [--sep_file]
                          [--is_sg]

optional arguments:
  -h, --help   show this help message and exit
  --path PATH  Path to the training data (default=./data).
  --name NAME  Name of training set (default=train_set.txt)
  --sep_file   Flag to indicate if separated files of train set should be
               created (default=False).
  --is_sg      Flag to indicate which model to use (default=False).


check_example.py [-h] [--test_path TEST_PATH]
                        [--results_path RESULTS_PATH] [--example EXAMPLE]

optional arguments:
  -h, --help            show this help message and exit
  --test_path TEST_PATH
                        Path to the file where analogy test examples are
                        stored (default=src/data/analogy_test.csv).
  --results_path RESULTS_PATH
                        Path to the json file where the analogy test results
                        are stored (default=results/sg_test2.json).
  --example EXAMPLE     An example number for inspection (default=0).
