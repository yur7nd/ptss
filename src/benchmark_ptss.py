import os
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm


def fetch_embeddings(_path):
    mod = torch.load(_path, map_location=torch.device('cuda:0'))
    try:
        _ents = mod['model'][0]['_entity_embedder.embeddings.weight'].cpu()
        _rels = mod['model'][0]['_relation_embedder.embeddings.weight'].cpu()
    except KeyError:
        _ents = mod['model'][0]['_entity_embedder._embeddings.weight'].cpu()
        _rels = mod['model'][0]['_relation_embedder._embeddings.weight'].cpu()
    return _ents, _rels


def get_triples(_ds):
    _df = pd.read_csv('ptss-benchmarks/triple_benchmarks_{}.csv'.format(_ds))
    print('Found {} benchmark triples for {}'.format(len(_df), _ds))
    return _df


def score_triples(_t1, _t2, _ent_embs, _rel_embs):
    h1 = _ent_embs[_t1[0]].unsqueeze(0)
    h2 = _ent_embs[_t2[0]].unsqueeze(0)
    r1 = _ent_embs[_t1[1]].unsqueeze(0)
    r2 = _ent_embs[_t2[1]].unsqueeze(0)
    t1 = _ent_embs[_t1[2]].unsqueeze(0)
    t2 = _ent_embs[_t2[2]].unsqueeze(0)
    h_sim = torch.cosine_similarity(h1, h2, ).detach().numpy()
    r_sim = torch.cosine_similarity(r1, r2).detach().numpy()
    t_sim = torch.cosine_similarity(t1, t2).detach().numpy()
    _score = np.mean([h_sim, r_sim, t_sim])
    return _score


def apply_scoring(_df, _ent_embs, _rel_embs):
    _all_scores = []
    for idx, vals in tqdm(_df.iterrows()):
        t1 = [vals['true_head'], vals['true_rel'], vals['true_tail']]
        t2 = [vals['head_idx'], vals['rel_idx'], vals['tail_idx']]
        sim = score_triples(t1, t2, _ent_embs, _rel_embs)
        _all_scores.append(sim)
    _df['sim_score'] = _all_scores
    return _df


if __name__ == "__main__":
    ds = ['wnrr', 'fb15k-237']
    model_paths = ['pytorch-models/{}-complex.pt',
                   'pytorch-models/{}-conve.pt',
                   'pytorch-models/{}-distmult.pt',
                   'pytorch-models/{}-rescal.pt',
                   'pytorch-models/{}-rotate.pt',
                   'pytorch-models/{}-transe.pt'
                   ]
    for d in ds:
        trip_df = get_triples(d)
        for mt in model_paths:
            mn = mt.split('/')[1].split('-')[1].split('.')[0]
            print('Running {}'.format(mn))
            ent_embs, rel_embs = fetch_embeddings(mt.format(d))
            df_out = apply_scoring(trip_df, ent_embs, rel_embs)
            df_out.to_csv('ptss-benchmarks/triple_scores_{}_{}.csv'.format(d, mn))
            print(df_out.head())

