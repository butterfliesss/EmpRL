# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
from __future__ import print_function

from nltk.tokenize import word_tokenize

def calc_distinct_n(n, candidates, print_score: bool = False): #True
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]

    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i : i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16) *100

    if print_score:
        print(f"***** nltk_tokenizer_cem_Distinct-{n}: {score*100} *****")
    return score

def calc_distinct(candidates, print_score: bool = False):
    scores = []
    for i in range(2):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)

    return scores

def cem_dist(hyp_list):
    cands = []
    for i in range(len(hyp_list)):
        hyps = hyp_list[i].strip()
        cands.append(hyps)
    dist_1, dist_2 = calc_distinct(cands)

    return dist_1, dist_2