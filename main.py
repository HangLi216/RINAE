
from sklearn.cross_validation import train_test_split
import gensim.utils as ut
import scipy.io as sio
import argparse
import numpy as np
from Embed import Auto_Embed
import Evalue
from con2vec import con2vec

def parse_args():
    ##
    # Parses the arguments.
    # #
    parser = argparse.ArgumentParser(description="Run")

    parser.add_argument('--embed_size', type=int, default=100,
                        help='represent dimension')

    parser.add_argument('--input_path', nargs='?', default="dataset/M10/M10.mat",
                        help='network path')

    parser.add_argument('--group_path', nargs='?', default="dataset/M10/M10_label.mat",
                        help='network group path')

    parser.add_argument('--ouput_save', nargs='?', default="dataset/M10",
                        help='Embeddings path')

    parser.add_argument('--window', type=int, default=8,
                        help='Context size for optimization.')

    parser.add_argument('--input_size', default=10310, type=int,
                        help='number of nodes')

    parser.add_argument('--min_count', type=int, default=2,
                        help='ignore all words with total frequency lower than this.')

    parser.add_argument('--passes', type=int, default=10,
                        help='Number of epochs in doc2vec')

    parser.add_argument('--dm', type=int, default=0,
                        help='defines the training algorithm of doc2vec')

    parser.add_argument('--size', dest='size', default=100,
                        help='the dimensionality of the text feature vectors.')
    parser.add_argument('--cores', dest='cores', default=4,help='Number of parallel workers')

    parser.add_argument('--doc_dir', dest='doc_dir',default='dataset/M10/docs.txt',
                        help='')
    parser.set_defaults(directed=False)

    return parser.parse_args()

##
#combine feature together
# #
def get_network(dic,doc_model,input_size=100):
    data = sio.loadmat(dic)
    network = data['network']
    vecs = [doc_model.docvecs[ut.to_unicode(str(j))] for j in range(input_size)]
    network=np.concatenate([network, vecs], axis=1)
    return network

def get_group(group_path):
    data = sio.loadmat(group_path)
    group = data['label']
    return group

def main(args):

    doc= con2vec(doc_dir=args.doc_dir, workers=args.cores, size=args.size, dm=args.dm,
                 window=args.window, passes=args.passes, min_count=args.min_count)

    network =get_network(args.input_path,doc.doc_model,args.input_size)

    embed_model=Auto_Embed(input_size=args.input_size+args.size,hid_dim1=500,hid_dim2=300,
               embed_size=args.embed_size,decay=1e-4,bias=True)

    train, test = train_test_split(network, train_size=0.7, random_state=6)

    embed_model.training_model(train,nb_epoch=8,batch_size=1000,optimizer='rmsprop',validation_split=0.3)

    embed=embed_model.get_embed(network)

    np.save(args.ouput_save+'/'+str(args.embed_size)+'emb',embed)
    # embed=np.load(args.ouput_save+'/'+str(args.embed_size)+'emb.npy')

    group=get_group(args.group_path)

    Our_macro_f1, Our_micro_f1=Evalue.evaluation(embed, group, classifierStr='SVM', train_size=0.7, seed=6)
    print('Our Classification macro_f1=%f, micro_f1=%f' % (Our_macro_f1, Our_micro_f1))

    doc_macro_f1, doc_micro_f1=Evalue.doc_evaluation(doc.doc_model, group, classifierStr='SVM', train_size=0.8, seed=6)
    print('doc Classification macro_f1=%f, micro_f1=%f' % (doc_macro_f1, doc_micro_f1))


if __name__ == "__main__":
    args = parse_args()
    main(args)
