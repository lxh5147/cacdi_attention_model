'''
Created on Jul 13, 2016

@author: lxh5147
'''
import numpy as np

def _get_dim(tree):
    if isinstance(tree, list):
        return 1 + _get_dim(tree[0])
    elif isinstance(tree, np.ndarray):
        return len(tree.shape)
    else:
        return 0

def _get_sequence_inputs(tree, inputs, mask_value=-1):
    # reviews, sentences, words
    ndim = _get_dim(tree)
    assert ndim >=2

    if ndim == 2:
        #sentences, words
        max_length = 0
        for s in tree:
            if max_length<len(s):
                max_length = len(s)
        input = np.zeros((len(tree),max_length),dtype=np.int)+ mask_value
        for i in xrange(len(tree)):
            for j in xrange(len(tree[i])):
                input[i][j]=tree[i][j]
        inputs.append(input)
    else:
        max_length = 0
        for r in tree:
            if max_length < len(r):
                max_length = len(r)
        input = np.zeros((len(tree),max_length),dtype=np.int) + mask_value
        sub_tree = []
        for i in xrange(len(tree)):
            for j in xrange(len(tree[i])):
                input[i][j] = len(sub_tree)
                sub_tree.append(tree[i][j])
        inputs.append(input)
        _get_sequence_inputs(sub_tree,inputs,mask_value)


def get_sequence_inputs(tree,  mask_value=-1):
    # reviews, sentences, words
    inputs = []
    _get_sequence_inputs(tree, inputs, mask_value)
    return inputs