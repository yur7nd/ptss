from train import run_triple_fitting


def experiments():
    mods = ['complex', 'conve', 'distmult',
            'rescal', 'rotate', 'transe']
    combs = ['ht', 'had', 'avg', 'l1', 'l2']
    ds = ['wnrr', 'fb15k-237']
    for d in ds:
        for m in mods:
            for c in combs:
                run_triple_fitting(m, c, d)


if __name__ == "__main__":
    experiments()

