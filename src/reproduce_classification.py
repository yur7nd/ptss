import pandas as pd
import numpy as np

from evaluate import load_space, measure_clusters, \
    get_relational_labels


def experiments(_t2v, _dup):
    if _t2v:
        ds = ['wnrr', 'fb15k-237']
        dims = [32, 64, 128, 256, 512, 1024]
        print('Loading triple2vec')
        all = []
        for _ds in ds:
            for _d in dims:
                el, eh = load_space(_ds, _d, 10, 't2v', 'edge', _dup)
                all.append(['t2v', _ds, _d, el, eh])
        out = pd.DataFrame(all)
        print(out)
        if _dup:
            v = 'restricted'
        else:
            v = 'all'
        out.to_csv('outputs/edge_classification_t2v_{}.csv'.format(v), index=False)

    else:
        mods = ['complex', 'conve', 'distmult',
                'rescal', 'rotate', 'transe']
        combs = ['ht', 'had', 'avg', 'l1', 'l2']
        ds = ['wnrr', 'fb15k-237']
        all = []
        for m in mods:
            for d in ds:
                for c in combs:
                    full_name = '{}-{}-{}'.format(d, m, c)
                    el, eh = load_space(d, 0, 0, full_name, 'edge', _dup)
                    all.append([d, m, c, el, eh])
        final = pd.DataFrame(all)
        print(final)
        if _dup:
            v = 'restricted'
        else:
            v = 'all'
        final.to_csv('outputs/edge_classification_{}.csv'.format(v), index=False)


if __name__ == "__main__":
    experiments(False, True)
    experiments(False, False)
    experiments(True, True)
    experiments(True, False)

