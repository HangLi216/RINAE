Keras implementation of "Effective Representing of Information Network by Variational Autoencoder."

Requirement:

Cython==0.21.2
gensim==0.12.0
numpy==1.9.1
pandas==0.17.0
patsy==0.4.0
scikit-learn==0.15.2
scipy==0.15.1
six==1.9.0
wheel==0.26.0
psutil>=2.1.1
futures>=2.1.6
argparse>=1.2.1
deepwalk==1.0.2
tensorflow==0.11.0
keras==1.1.1

parameter and file describe
The 'input_path' read the data from the directory 'dataset/M10', the directory should contain three files:
	docs.txt -- text document for each node, one line for one node eg. ID text information
	labels.mat -- class label for each node, one line for one node eg. label
	adjedges.mat -- adjacency vector of each node, one line for one node.


	'embed_size': represent dimension  eg.100,200
	'input_path': network path eg.dataset/M10/M10.mat
	'group_path': network group path eg.dataset/M10/M10_label.mat
	'ouput_save': Embedding save path eg. dataset/M10
	'window':     Context size for optimization.
	'input_size': number of nodes
	'min_count' : minimum number of counts for words.
	'passes'    : Number of epochs in doc2vec
	'dm'        : defines the training algorithm of doc2vec
	'size'      : the dimensionality of the text feature vectors
	'cores'     : Number of parallel workers
	'doc_dir'   : text document path
