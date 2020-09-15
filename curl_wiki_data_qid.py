# -*- coding: utf-8 -*-
import requests
import pdb
import xml.etree.ElementTree as ET
import time
import json
import numpy as np

def completion():

    encoded_idx = np.zeros(12554)
    with open("interpolation/wiki/entity2id_name.txt", 'r', encoding="utf-8") as f:
        for line in f:
            # print(line.encode("utf-8"))
            try:
                ename, id = line.split('\t')
            except:
                pdb.set_trace()
            id = int(id)
            encoded_idx[id] = 1

    # pdb.set_trace()
    with open("interpolation/wiki/entity2id.txt", 'r') as fr,\
            open("interpolation/wiki/entity2id_name.txt", 'a+', encoding="utf-8") as fw:
        for line in fr:
            qid, id = line.split('\t')
            id = int(id)
            if encoded_idx[id] == 0:
                # print("{}\t{}".format(qid, id))
                try:
                    response = requests.get(
                        'https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids={}&format=json'.format(
                            qid))
                    r_json = json.loads(response.text.encode('utf-8'))
                    # pdb.set_trace()
                    answer = r_json['entities'][qid]['labels']['en']['value']
                    fw.write('{}\t{}\n'.format(answer, id))
                    time.sleep(0.1)
                except:
                    print(qid)

def shared_encoding(fr, fw):
    for line in fr:
        try:
            qid, id = line.split('\t')
            r = requests.get(
                'https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids={}&languages=en&format=json'.format(
                    qid))

            # print(r.text)
            r_json = json.loads(r.text.encode('utf-8'))
            # r_json = r.json()
            answer = r_json['entities'][qid]['labels']['en']['value']
            print(answer.encode("utf-8"))

            fw.write('{}\t{}'.format(answer, id))
            time.sleep(1)
        except:
            print(qid)

def convert_entity():

    with open("interpolation/wiki/entity2id.txt", 'r') as fr,\
            open("interpolation/wiki/entity2id_name.txt", 'w', encoding="utf-8") as fw:
        shared_encoding(fr, fw)

def convert_relations():

    with open("interpolation/wiki/relation2id.txt", 'r') as fr,\
            open("interpolation/wiki/relation2id_name.txt", 'w', encoding="utf-8") as fw:
        shared_encoding(fr, fw)

if __name__ == '__main__':
    # qids = []
    convert_relations()