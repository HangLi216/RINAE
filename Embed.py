from keras.layers import Input, Dense, Lambda, Dropout,merge
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras import objectives
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy.io as sio



class Auto_Embed:
    def __init__(self,input_size,hid_dim1,hid_dim2,embed_size=100,decay=1e-4,bias=True):
        self.input_size=input_size
        self.hid_dim1=hid_dim1
        self.hid_dim2=hid_dim2
        self.embed_size=embed_size
        self.decay=decay
        self.bias=bias

        self.x = Input(shape=(self.input_size,))
        self.h_encoded1 = Dense(self.hid_dim1, W_regularizer=l2(self.decay), b_regularizer=l2(self.decay),
                           bias=self.bias, activation='tanh')(self.x)
        self.h_encoded = Dense(self.hid_dim2, W_regularizer=l2(self.decay), b_regularizer=l2(self.decay),
                          bias=self.bias, activation='tanh')(self.h_encoded1)

        self.z_mean = Dense(self.embed_size, W_regularizer=l2(self.decay), b_regularizer=l2(self.decay),
                            bias=self.bias)(self.h_encoded)
        self.z_log_var = Dense(self.embed_size, W_regularizer=l2(self.decay), b_regularizer=l2(self.decay),
                               bias=self.bias)(self.h_encoded)
        self.z = Lambda(self.sampling, output_shape=(self.embed_size,))([self.z_mean, self.z_log_var])

        self.z_mean2 = Dense(self.embed_size, W_regularizer=l2(self.decay), b_regularizer=l2(self.decay),
                            bias=self.bias)(self.h_encoded)
        self.z_log_var2 = Dense(self.embed_size, W_regularizer=l2(self.decay), b_regularizer=l2(self.decay),
                               bias=self.bias)(self.h_encoded)
        self.z2 = Lambda(self.sampling, output_shape=(self.embed_size,))([self.z_mean2, self.z_log_var2])

        self.con = merge([self.z, self.z2], mode='sum')

        self.h_decoder = Dense(self.hid_dim2, W_regularizer=l2(self.decay), b_regularizer=l2(self.decay),
                          bias=self.bias, activation='tanh')(self.con)

        self.decoder_h = Dense(self.hid_dim1, W_regularizer=l2(self.decay), b_regularizer=l2(self.decay),
                          bias=self.bias, activation='tanh')(self.h_decoder)

        self.x_hat = Dense(self.input_size, W_regularizer=l2(self.decay), b_regularizer=l2(self.decay),
                      bias=self.bias, activation='sigmoid')(self.decoder_h)


    def training_model(self,train,nb_epoch,batch_size,optimizer,validation_split):
        model = Model(input=self.x, output=self.x_hat)
        model.compile(optimizer=optimizer, loss=self.loss)
        model.fit(train, train,
                  shuffle=False,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  validation_split=validation_split)
        self.model=model

    def sampling(self,arg):
        epsilon = K.random_normal(shape=(self.embed_size,), mean=0., std=1.0)
        return arg[0] + K.exp(arg[1] / 2) * epsilon

    def loss(self,inputs, outputs):
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        kl_loss2 = - 0.5 * K.sum(1 + self.z_log_var2 - K.square(self.z_mean2) - K.exp(self.z_log_var2), axis=-1)
        xent_loss = self.input_size * objectives.binary_crossentropy(self.x, self.x_hat)
        return xent_loss + kl_loss+kl_loss2

    def encoder(self):
        model = Model(input=self.x, output=self.con)
        return model

    def decoder(self):
        model = Model(input=self.z, output=self.x_hat)
        return model

    def get_embed(self,network):
        encoder=self.encoder()
        embedding = encoder.predict(network)
        return embedding