# import pkg_resources
import os
import errno
import math
from pathlib import Path
import pickle
import re
import sys
import numpy as np
import pdb


def get_id2text():
    ent2id = {}
    rel2id = {}
    with open(os.path.join(path, "entities"), "r") as f:
        for line in f.readlines():
            try:
                id, qid, text = line.strip().split('\t')
            except:
                text = ''
                # pdb.set_trace()
            ent2id[int(id)] = text

    with open(os.path.join(path, "predicates"), "r") as f:
        for line in f.readlines():
            id, qid, text = line.strip().split('\t')
            rel2id[int(id)] = text

    return ent2id, rel2id

def get_be(begin, end):
    begin = re.search(r'([+-]\d+)-(\d+)-(\d+)', begin)
    end = re.search(r'([+-]\d+)-(\d+)-(\d+)', end)
    if begin is None:
        begin = (-math.inf, 0, 0)
    else:
        begin = (int(begin.group(1)), 0, 0)

    if end is None:
        end = (math.inf, 0, 0)
    else:
        end = (int(end.group(1)), 0, 0)

    return begin, end


def prepare_dataset_rels(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(type)\t(timestamp)\n
    Maps each entity, relation+type and timestamp to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations, timestamps = set(), set(), set()
    begin_min = 1959
    end_max = 2019
    for f in files:
        total_count = 0
        non_temp = 0
        temp = 0
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            v = line.strip().split('\t')
            lhs, rel, rhs, begin, end = v
            begin, end = get_be(begin, end)
            # if int(rel) == 265:
            # if begin[0] != math.inf and begin[0] > 2020:

            if end[0] < begin_min or (begin[0] > end_max):
                continue
                # print("{}\t{}\t{}\t{}\t{}".format(ent2text[int(lhs)], rel2text[int(rel)], ent2text[int(rhs)], begin[0], end[0]))

            begin = max(begin_min, begin[0])
            end = min(end_max, end[0])

            if begin <= end:
                total_count += end - begin + 1
            if begin == begin_min and end == end_max:
                non_temp += end - begin + 1
            else:
                temp += end - begin + 1
            timestamps.add(begin)
            timestamps.add(end)
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
        to_read.close()
        print("Total quadruples: {} in {} set".format(total_count, f))
        print("Non-temporal: {}".format(non_temp))
        print("Temporal: {}".format(temp))
    # exit()
    print(f"{len(timestamps)} timestamps")

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}

    # we need to sort timestamps and associate them to actual dates
    all_ts = sorted(timestamps)
    timestamps_to_id = {x: i for (i, x) in enumerate(all_ts)}
    # print(timestamps_to_id)

    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)

    if not os.path.isdir(os.path.join(DATA_PATH, name)):
        os.makedirs(os.path.join(DATA_PATH, name))

    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'ts_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'wb')
        pickle.dump(dic, ff)
        ff.close()

    # dump the time differences between timestamps for continuity regularizer
    # ignores number of days in a month but who cares
    # ts_to_int = [x[0] * 365 + x[1] * 30 + x[2] for x in all_ts]
    # ts_to_int = [x[0] for x in all_ts]
    # ts = np.array(ts_to_int, dtype='float')
    # diffs = ts[1:] - ts[:-1]  # remove last timestamp from time diffs. it's not really a timestamp
    # out = open(os.path.join(DATA_PATH, name, 'ts_diffs.pickle'), 'wb')
    # pickle.dump(diffs, out)
    # out.close()

    # map train/test/valid with the ids
    for f in files:

        total_count = 0
        file_path = os.path.join(path, f)

        out_f = os.path.join(DATA_PATH, name, "{}.txt".format(f))
        to_read = open(file_path, 'r')
        to_write = open(out_f, 'w')
        ignore = 0
        total = 0
        full_intervals = 0
        half_intervals = 0
        point = 0
        for line in to_read.readlines():
            v = line.strip().split('\t')
            lhs, rel, rhs, begin, end = v
            begin, end = get_be(begin, end)
            total += 1

            if end[0] < begin_min or begin[0] > end_max:
                continue
            begin = max(begin_min, begin[0])
            end = min(end_max, end[0])

            begin = timestamps_to_id[begin]
            end = timestamps_to_id[end]
            # print(v)
            # print(begin, end)
            # pdb.set_trace()
            if begin > end:
                ignore += 1
                continue

            lhs = entities_to_id[lhs]
            rel = relations_to_id[rel]
            rhs = entities_to_id[rhs]

            for t in range(begin, end + 1):
                # pdb.set_trace()
                total_count += 1
                to_write.write("{}\t{}\t{}\t{}\n".format(lhs, rel, rhs, t))
        # out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        # pickle.dump(np.array(examples).astype('uint64'), out)
        # out.close()
        print(f"Ignored {ignore} events.")
        print(f"Total : {total} // Full : {full_intervals} // Half : {half_intervals} // Point : {point}")
        to_read.close()
        to_write.close()


if __name__ == "__main__":

    DATA_PATH = os.path.join('..','interpolation')
    path = os.path.join('..','raw', 'wikidata')
    ent2text, rel2text = get_id2text()
    print("Preparing dataset {}".format('wikidata'))
    prepare_dataset_rels(path, 'wikidata')
