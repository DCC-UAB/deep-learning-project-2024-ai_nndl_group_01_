import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from params import *

def build_phoc(words, phoc_unigrams, unigram_levels,
               bigram_levels=None, phoc_bigrams=None,
               split_character=None):
    '''
    Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).
    Args:
        word (str): word to calculate descriptor for
        phoc_unigrams (str): string of all unigrams to use in the PHOC
        unigram_levels (list of int): the levels for the unigrams in PHOC
        phoc_bigrams (list of str): list of bigrams to be used in the PHOC
        phoc_bigram_levls (list of int): the levels of the bigrams in the PHOC
        split_character (str): special character to split the word strings into characters
        on_unknown_unigram (str): What to do if a unigram appearing in a word
            is not among the supplied phoc_unigrams. Possible: 'warn', 'error'
    Returns:
        the PHOC for the given word
    '''
    # prepare output matrix
    phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)
    if phoc_bigrams is not None:
        phoc_size += len(phoc_bigrams) * np.sum(bigram_levels)
    phocs = np.zeros((len(words), phoc_size))

    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k + 1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]

    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(phoc_unigrams)}

    # iterate through all the words
    for word_index, word in enumerate(words):
        """if '0' in word or '1' in word or '2' in word or '3' in word or '4' in word or '5' in word or '6' in word or '7' in word or '7' in word or '8' in word or '9' in word:
            continue"""
        if split_character is not None:
            word = word.split(split_character)
        n = len(word)
        for index, char in enumerate(word):
            char_occ = occupancy(index, n)        
            char_index = char_indices[char]

            for level in unigram_levels:
                for region in range(level):
                    region_occ = occupancy(region, level)
                    if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                        feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(
                            phoc_unigrams) + region * len(phoc_unigrams) + char_index
                        phocs[word_index, feat_vec_index] = 1

        # add bigrams
        if phoc_bigrams is not None:
            ngram_features = np.zeros(len(phoc_bigrams) * np.sum(bigram_levels))
            ngram_occupancy = lambda k, n: [float(k) / n, float(k + 2) / n]

            for i in range(n - 1):
                ngram = word[i:i + 2]
                phoc_dict = {k: v for v, k in enumerate(phoc_bigrams)}
                if phoc_dict.get(ngram, 666) == 666:
                    continue
                occ = ngram_occupancy(i, n)

                for level in bigram_levels:
                    for region in range(level):
                        region_occ = occupancy(region, level)
                        overlap_size = size(overlap(occ, region_occ)) / size(occ)
                        if overlap_size >= 0.5:
                            ngram_features[region * len(phoc_bigrams) + phoc_dict[ngram]] = 1
            phocs[word_index, -ngram_features.shape[0]:] = ngram_features

    return phocs

def phoc(raw_word):
    '''

    :param raw_word: string of word to be converted
    :return: phoc representation as a np.array (1,604)
    '''
    if type(raw_word) == type([]):
        word = [w.lower() for w in raw_word]
    else:
        word =[raw_word]
        word_lowercase = word[0].lower()
        word = [word_lowercase]
    phoc_unigrams = '0123456789abcdefghijklmnopqrstuvwxyz'
    unigram_levels = [2,3,4,5]
    bigram_levels=[]
    bigram_levels.append(2)
    
    phoc_bigrams = []
    i = 0
    with open(bigrams_file,'r') as f:
        for line in f:
            a = line.split()
            phoc_bigrams.append(a[0].lower())
            i = i +1
            if i >= 50:break

    
    qry_phocs = build_phoc(words = word, phoc_unigrams = phoc_unigrams, unigram_levels = unigram_levels,
                           bigram_levels = bigram_levels, phoc_bigrams = phoc_bigrams)
    qry_phocs = build_phoc(words = word, phoc_unigrams = phoc_unigrams, unigram_levels = unigram_levels)
    
    return qry_phocs