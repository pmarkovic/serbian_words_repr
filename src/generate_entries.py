import json
from collections import defaultdict
from gensim.models import Word2Vec


TOPICS = ["sport", "colours", "countries", "cities", "food-drink", 
          "names", "family-rel", "vehicles", "astro", "religion"]


def get_entries():
    entries = []

    with open("./data/entries.txt", 'r') as txt_file:
        for line in txt_file:
            entries.append(line.split(',')[0])

    return entries


def do_fix(ext_entries):
    del(ext_entries["lišće"])

    fixes = [("teniserka", "sport"), ("slatka", "food-drink"), ("tamna", "colours"), ("svetlo", "colours"), ("sunce", "astro"), ("salata", "food-drink"),
             ("zemlja", "astro"), ("ljubica", "names"), ("otac", "family-rel"), ("iguman", "religion"), ("sveštenik", "religion"), ("starac", "family-rel"),
             ("monah", "religion"), ("oganj", "religion"), ("hristovo", "religion"), ("gospod", "religion")]


    for fix in fixes:
        ext_entries[fix[0]] = {fix[1]}

    return {k: v.pop() for k, v in ext_entries.items()}


def expand_entries(entries):
    gen_model = Word2Vec.load("../weights/gen_sg300.txt")
    ext_entries = defaultdict(set)

    for i, entry in enumerate(entries):
        curr_topic = TOPICS.pop(0) if i % 3 == 0 else curr_topic

        ext_entries[entry].add(curr_topic)

        for similar in gen_model.wv.most_similar(entry, topn=20):
            ext_entries[similar[0]].add(curr_topic)

    final_result = do_fix(ext_entries)
    
    return final_result


if __name__ == "__main__":
    entries = get_entries()

    ext_entries = expand_entries(entries)

    with open("entries.json", 'w') as json_file:
        json.dump(ext_entries, json_file, indent=2)
