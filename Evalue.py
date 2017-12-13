from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import gensim.utils as ut
from sklearn.cross_validation import train_test_split
from gensim.models.doc2vec import Doc2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


def evaluation(embed, group, classifierStr='SVM', normalize=0,train_size=0.5,seed=6):

    train_vec, test_vec, train_y, test_y = train_test_split(embed, group, train_size=train_size, random_state=seed)

    if classifierStr == 'KNN':
        print('Training NN classifier...')
        classifier = KNeighborsClassifier(n_neighbors=1)
    elif classifierStr=='SVM':
        print('Training SVM classifier...')

        classifier = LinearSVC()
    else:
        print('Training Logistic...')

        classifier = LogisticRegression()

    if(normalize == 1):
        print('Normalize data')
        allvec = list(train_vec)
        allvec.extend(test_vec)
        allvec_normalized = preprocessing.normalize(allvec, norm='l2', axis=1)
        train_vec = allvec_normalized[0:len(train_y)]
        test_vec = allvec_normalized[len(train_y):]

    classifier.fit(train_vec, train_y)
    y_pred = classifier.predict(test_vec)
    cm = confusion_matrix(test_y, y_pred)

    print "confusion_matrix"
    print(cm)

    macro_f1 = f1_score(test_y, y_pred,pos_label=None, average='macro')
    micro_f1 = f1_score(test_y, y_pred,pos_label=None, average='micro')

    per = len(train_y) * 1.0 /(len(test_y)+len(train_y))
    print('Classification method:'+classifierStr+'(train, test, Training_Percent): (%d, %d, %f)' % (len(train_y),len(test_y), per ))
    print('Classification macro_f1=%f, micro_f1=%f' % (macro_f1, micro_f1))

    return macro_f1, micro_f1


def doc_evaluation(doc_model, group, classifierStr='SVM', normalize=0,train_size=0.5,seed=6):

    dim=int(len(group))
    vecs = [doc_model.docvecs[ut.to_unicode(str(j))] for j in range(dim)]
    macro_f1, micro_f1=evaluation(vecs,group,classifierStr, normalize,train_size,seed)

    return macro_f1, micro_f1