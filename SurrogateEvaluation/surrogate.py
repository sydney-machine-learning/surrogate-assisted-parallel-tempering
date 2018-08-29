# !/usr/bin/python
from __future__ import division
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
# Import keras methods and classes
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.objectives import MSE, MAE
from keras.callbacks import EarlyStopping
# Import train and test split method from Scikit-Learn
from sklearn.model_selection import train_test_split

############################################################################
# Surrogate model Class
class Surrogate(object):
    def __init__(self, name, x_train, x_test, y_train, y_test, ratio, path, dropout):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.path = path
        # set display = False if you do not wish to print the min and max values of y
        self.min_Y, self.max_Y = self.read_min_max(display=False)
        self.num_features = x_train.shape[1]
        self.train_size = x_train.shape[0]
        self.test_size = x_test.shape[0]
        self.krnn = self.create_model(dropout)
        self.batch_ratio = np.round(ratio, decimals=2)
        self.dropout = np.round(dropout, decimals=2)
        num_batches = int(1.0/ratio)
        self.results_path = './results_'+name+'_'+str(self.batch_ratio)+'_'+str(self.dropout)
        self.make_directory(self.results_path)
        self.create_train_batches(num_batches)

    def create_model(self, dropout):
        krnn = Sequential()
        krnn.add(Dense(64, input_dim=self.num_features, kernel_initializer='uniform', activation='relu'))
        krnn.add(Dropout(dropout))
        krnn.add(Dense(35, kernel_initializer='uniform', activation='relu'))
        krnn.add(Dropout(dropout))
        krnn.add(Dense(20, kernel_initializer='uniform', activation='relu'))
        krnn.add(Dropout(dropout))
        krnn.add(Dense(1, kernel_initializer ='uniform', activation='sigmoid'))
        return krnn

    def create_train_batches(self, num_batches):
        self.num_batches = num_batches
        self.x_train_batches = []
        self.y_train_batches = []
        splitsize = 1.0/num_batches*self.train_size
        for index in range(self.num_batches):
            self.x_train_batches.append(self.x_train[ int(round(index*splitsize)) : int(round((index+1)*splitsize)), :])
            self.y_train_batches.append(self.y_train[ int(round(index*splitsize)) : int(round((index+1)*splitsize))])

    def train_model(self,):
        model_signature = 1
        desired_train = self.y_train*(self.max_Y - self.min_Y) + self.min_Y
        desired_test = self.y_test*(self.max_Y - self.min_Y) + self.min_Y
        self.make_directory(self.results_path+'/prediction')
        self.make_directory(self.results_path+'/loss')
        with open(self.results_path+'/results.txt', 'w') as file:
            for index in range(self.num_batches):
                early_stopping = EarlyStopping(monitor='val_loss', patience=20)
                self.krnn.compile(loss='mse', optimizer='adam', metrics=['mse'])
                train_log = self.krnn.fit(self.x_train_batches[index], self.y_train_batches[index], batch_size=10, epochs=100, validation_split=0.1, verbose=2, callbacks=[early_stopping])
                scores = self.krnn.evaluate(self.x_test, self.y_test.ravel(), verbose = 0)
                print("Test metric %s: %.5f" % (self.krnn.metrics_names[1], scores[1]))
                self.krnn.save(self.results_path+'/model_krnn_{}_.h5'.format(model_signature))
                print "Saved model to disk with model signature", model_signature
                file.write('Model Signature: '+str(model_signature)+' MSE: '+str(scores[1])+'\n')

                # Plot results of training
                plt.plot(train_log.history["loss"], label="loss")
                plt.plot(train_log.history["val_loss"], label="val_loss")
                plt.legend()
                plt.savefig(self.results_path+'/loss/{}_0.png'.format(model_signature))
                plt.clf()
                self.test_model(self.x_train, desired_train, model_signature, name='train')
                self.test_model(self.x_test, desired_test, model_signature, name='test')
                model_signature += 1

    def test_model(self, data, desired, model_signature, name):
        krnn_prediction = self.krnn.predict(data).reshape(data.shape[0])
        prediction = krnn_prediction*(self.max_Y - self.min_Y) + self.min_Y
        plt.plot(prediction, label='prediction')
        plt.plot(desired, label='desired')
        plt.title('Surrogate Prediction Plot')
        plt.legend()
        plt.savefig(self.results_path+'/prediction/{}_0_{}_prediction.png'.format(model_signature,name))
        plt.clf()

    def read_min_max(self, display=True):
        with open(path+'/minmax.txt', 'r') as file:
            min_Y = float(file.readline())
            max_Y = float(file.readline())
        if display == True:
            print "min_y: {}  max_y: {}".format(min_Y, max_Y)
        return min_Y, max_Y

    @staticmethod
    def make_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


##########################################################################
if __name__ == '__main__':
    problems = ['iris', 'cancer' ]
    batch_ratios = np.linspace(0.1, 0.5, 5)
    dropout_values = np.linspace(0.0, 0.5, 5)
    for ratio in batch_ratios:
        for dropout in dropout_values[:1]:
            for problem in problems:
                # Parameters for data
                problem_paths = {'iris':'iris', 'cancer':'cancer' }
                num_batches = 5
                path = problem_paths[problem]
                #################################
                # Load train and test data
                x_train = np.genfromtxt(path+'/X_train.csv', delimiter=' ')
                y_train = np.genfromtxt(path+'/Y_train.csv', delimiter=' ')
                x_test = np.genfromtxt(path+'/X_test.csv', delimiter=' ')
                y_test = np.genfromtxt(path+'/Y_test.csv', delimiter=' ')
                ################################
                # Create Surrogate class instance
                # path = 'results/'+problem+'_'+str(num_batches)+'/'
                surrogate_model = Surrogate(problem, x_train, x_test, y_train, y_test, ratio, path, dropout)
                ################################
                # Train model in batches
                surrogate_model.train_model()
                ################################
