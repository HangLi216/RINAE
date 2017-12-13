from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec, Word2Vec
from random import shuffle
import gensim
import gensim.utils as ut
import scipy.io as sio
import argparse
import numpy as np
from Embed import Auto_Embed
import Evalue

####################
#con2vec process
##############
class con2vec:
    def __init__(self,doc_dir,workers,size,dm,window,passes,min_count):
        self.doc_dir=doc_dir
        self.workers = workers
        self.size = size
        self.dm = dm
        self.window = window
        self.passes = passes
        self.min_count = min_count

        alldocs, allsentence = self.readNetworkData()

        self.doc_model = self.trainDoc2Vec(alldocs)

    def trainDoc2Vec(self, alldocs=None, buildvoc=1, dm_mean=0, hs=1, negative=5):
        doc_list = alldocs[:]  # for reshuffling per pass
        model = Doc2Vec(dm=self.dm, size=self.size, dm_mean=dm_mean, window=self.window,
                        hs=hs, negative=negative, min_count=self.min_count, workers=self.workers)  # PV-DBOW
        if buildvoc == 1:
            print('Building Vocabulary')
            model.build_vocab(doc_list)  # build vocabulate with words + nodeID

        for epoch in range(self.passes):
            # print('Iteration %d ....' % epoch)
            shuffle(doc_list)  # shuffling gets best results
            model.train(doc_list)

        return model

    ##
    # process content info
    # #
    def readNetworkData(self, stemmer=0):  # dir, directory of network dataset
        allindex = {}
        alldocs = []
        NetworkSentence = namedtuple('NetworkSentence', 'words tags index')
        with open(self.doc_dir) as f1:
            for l1 in f1:
                # tokens = ut.to_unicode(l1.lower()).split()
                if stemmer == 1:
                    l1 = gensim.parsing.stem_text(l1)
                else:
                    l1 = l1.lower()
                tokens = ut.to_unicode(l1).split()

                words = tokens[1:]
                tags = [tokens[0]]  # ID of each document, for doc2vec model
                index = len(alldocs)
                allindex[tokens[0]] = index  # A mapping from documentID to index, start from 0
                alldocs.append(NetworkSentence(words, tags, index))

        return alldocs, allindex








