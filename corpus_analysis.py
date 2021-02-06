import os
import logging
from lxml import etree
from collections import defaultdict


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", 
                    filename="app.log",
                    filemode="w",
                    level=logging.INFO, 
                    datefmt="%d-%b-%y %H:%M:%S")

def fun(iterator):
    sent = list()

    for i in iterator:
        for line in i.split('\n'):
            if len(line) > 1:
                sent.append(line.split('\t')[1])

    return len(sent), ' '.join(sent)

logging.info("Program started ...")

corpus_path = os.path.join(os.path.dirname(os.getcwd()), "corpus/srWaC1.1.01.xml")
domains = defaultdict(int)
sent_count = 0

context = etree.iterparse(corpus_path, events=("start", "end"))
corpus_id = None
for event, element in context:
    if event == "start":
        if element.tag == "corpus":
            corpus_id = element.attrib["id"]
        elif element.tag == 'p':
            domains[element.attrib["urldomain"]] += 1
        elif element.tag == 's':
            sent_count += 1
            print(fun(element.itertext()))
    elif event == "end":
        element.clear()

    if sent_count == 1000:
        break

print(f"Sent count: {sent_count}")
print(f"Corpus id: {corpus_id}")

logging.info("Program finished !!!")

