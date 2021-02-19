import os
import time
import logging
from lxml import etree


fence = '=' * 20
CORPORA = ["srWaC1.1.01.xml", "srWaC1.1.02.xml", "srWaC1.1.03.xml", 
            "srWaC1.1.04.xml", "srWaC1.1.05.xml", "srWaC1.1.06.xml"]
CORPUS_PATH = os.path.join(os.path.dirname(os.getcwd()), "corpus")
DATA_PATH = os.path.join(os.getcwd(), "data")
MIN_LENGTH = 6


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", 
                    filename="preprocessing.log",
                    filemode="w",
                    level=logging.INFO, 
                    datefmt="%d-%b-%y %H:%M:%S")

def parse(iterator):
    sent = list()

    for i in iterator:
        for line in i.split('\n'):
            tokens = line.split('\t')
            if len(tokens) > 1 and tokens[1].isalnum():
                sent.append(tokens[1])
    sent.append('. ')

    return len(sent), ' '.join(sent)

logging.info(f"{fence}Program started{fence}")
program_start_time = time.time()

for corpus in CORPORA:
    corpus_start_time = time.time()
    context = etree.iterparse(os.path.join(CORPUS_PATH, corpus), events=("start", "end"))
    corpus_id = corpus[:-4]
    data_dir = os.path.join(DATA_PATH, corpus_id)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    with open(f"{os.path.join(data_dir, corpus_id)}.txt", 'a') as txt_file:
        for event, element in context:
            if event == "start" and element.tag == 's':
                sent_len, sent = parse(element.itertext())
                if sent_len >= MIN_LENGTH:
                    txt_file.write(sent)
            elif event == "end":
                element.clear()
    
    logging.info(f"Corpus {corpus_id} time: {round(time.time() - corpus_start_time, 2)}s")

logging.info(f"Total time: {round(time.time() - program_start_time, 2)}s")
logging.info(f"{fence}Program finished{fence}")

