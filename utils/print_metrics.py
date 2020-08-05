import pickle
import sys
import pdb
import os

if __name__ == '__main__':
    fname = os.path.join(sys.argv[1], "metrics-per-snapshot.pt")
    with open(fname, 'rb') as fp:
        metrics = pickle.load(fp)

    for k, vs in metrics.items():
        print(k)
        # pdb.set_trace()
        if type(vs) == dict:
            for v in vs.values():
                print(v)
        else:
            print(vs)
        print()
