""" Feed Forward Network with Parallel Tempering for Multi-Core Systems"""
 
from __future__ import print_function, division
import multiprocessing
import os
import sys
import gc
import numpy as np
import random
import time
import operator
import math
import matplotlib as mpl
mpl.use('agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt  
import matplotlib.ticker as mtick
plt.rcParams['xtick.labelsize'] = 12 
plt.rcParams['ytick.labelsize'] = 12

params = {'legend.fontsize': 12,
          'legend.handlelength': 2}
plt.rcParams.update(params)

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.stats import multivariate_normal
from scipy.stats import norm
#import GPy
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import normalize
import pickle
#np.random.seed(1)

import io 
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.objectives import MSE, MAE
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.models import load_model

import sys
import time 


class Network:

	def __init__(self, Topo, Train, Test, learn_rate):
		self.Top = Topo  # NN topology [input, hidden, output]
		self.TrainData = Train
		self.TestData = Test
		self.lrate = learn_rate
		self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
		self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
		self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
		self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer
		self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
		self.out = np.zeros((1, self.Top[2]))  # output last layer
		self.pred_class = 0

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sampleEr(self, actualout):
		error = np.subtract(self.out, actualout)
		sqerror = np.sum(np.square(error)) / self.Top[2]
		return sqerror

	def ForwardPass(self, X):
		z1 = X.dot(self.W1) - self.B1
		self.hidout = self.sigmoid(z1)  # output of first hidden layer
		z2 = self.hidout.dot(self.W2) - self.B2
		self.out = self.sigmoid(z2)  # output second hidden layer

		self.pred_class = np.argmax(self.out)


		#print(self.pred_class, self.out, '  ---------------- out ')

	'''def BackwardPass(self, Input, desired):
		out_delta = (desired - self.out).dot(self.out.dot(1 - self.out))
		hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))
		print(self.B2.shape)
		self.W2 += (self.hidout.T.reshape(self.Top[1],1).dot(out_delta) * self.lrate)
		self.B2 += (-1 * self.lrate * out_delta)
		self.W1 += (Input.T.reshape(self.Top[0],1).dot(hid_delta) * self.lrate)
		self.B1 += (-1 * self.lrate * hid_delta)'''


 

	def BackwardPass(self, Input, desired): # since data outputs and number of output neuons have different orgnisation
		onehot = np.zeros((desired.size, self.Top[2]))
		onehot[np.arange(desired.size),int(desired)] = 1
		desired = onehot
		out_delta = (desired - self.out)*(self.out*(1 - self.out))
		hid_delta = np.dot(out_delta,self.W2.T) * (self.hidout * (1 - self.hidout))
		self.W2 += np.dot(self.hidout.T,(out_delta * self.lrate))
		self.B2 += (-1 * self.lrate * out_delta)
		Input = Input.reshape(1,self.Top[0])
		self.W1 += np.dot(Input.T,(hid_delta * self.lrate))
		self.B1 += (-1 * self.lrate * hid_delta)


	def decode(self, w):
		w_layer1size = self.Top[0] * self.Top[1]
		w_layer2size = self.Top[1] * self.Top[2]

		w_layer1 = w[0:w_layer1size]
		self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

		w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
		self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
		self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]].reshape(1,self.Top[1])
		self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]].reshape(1,self.Top[2])

 

	def encode(self):
		w1 = self.W1.ravel()
		w1 = w1.reshape(1,w1.shape[0])
		w2 = self.W2.ravel()
		w2 = w2.reshape(1,w2.shape[0])
		w = np.concatenate([w1.T, w2.T, self.B1.T, self.B2.T])
		w = w.reshape(-1)
		return w

	def softmax(self):
		prob = np.exp(self.out)/np.sum(np.exp(self.out))
		return prob
 


	def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

		self.decode(w)  # method to decode w into W1, W2, B1, B2.
		size = data.shape[0]

		Input = np.zeros((1, self.Top[0]))  # temp hold input
		Desired = np.zeros((1, self.Top[2]))
		fx = np.zeros(size)

		for i in range(0, depth):
			for i in range(0, size):
				pat = i
				Input = data[pat, 0:self.Top[0]]
				Desired = data[pat, self.Top[0]:]
				self.ForwardPass(Input)
				self.BackwardPass(Input, Desired)
		w_updated = self.encode()

		return  w_updated

	def evaluate_proposal(self, data, w ):  # BP with SGD (Stocastic BP)

		self.decode(w)  # method to decode w into W1, W2, B1, B2.
		size = data.shape[0]

		Input = np.zeros((1, self.Top[0]))  # temp hold input
		Desired = np.zeros((1, self.Top[2]))
		fx = np.zeros(size)
		prob = np.zeros((size,self.Top[2]))

		for i in range(0, size):  # to see what fx is produced by your current weight update
			Input = data[i, 0:self.Top[0]]
			self.ForwardPass(Input)
			fx[i] = self.pred_class
			prob[i] = self.softmax()

		#print(fx, 'fx')
		#print(prob, 'prob' )

		return fx, prob


class surrogate: #General Class for surrogate models for predicting likelihood given the weights

	def __init__(self, model, X, Y, min_X, max_X, min_Y , max_Y, path, save_surrogatedata):
		
		self.path = path + '/surrogate'
		indices = np.where(Y==np.inf)[0]
		X = np.delete(X, indices, axis=0)
		Y = np.delete(Y,indices, axis=0)
		self.model_signature = 0.0
		self.X = X
		self.Y = Y
		self.min_Y = min_Y
		self.max_Y =  max_Y
		self.min_X = min_X
		self.max_X = max_X 

		self.save_surrogatedata =  save_surrogatedata

		if model=="gp":
			self.model_id = 1
		elif model == "nn":
			self.model_id = 2
		elif model == "krnn":
			self.model_id = 3
			self.krnn = Sequential()
		else:
			print("Invalid Model!")

	def normalize(self, X):
		maxer = np.zeros((1,X.shape[1]))
		miner = np.ones((1,X.shape[1]))

		for i in range(X.shape[1]):
			maxer[0,i] = max(X[:,i])
			miner[0,i] = min(X[:,i])
			X[:,i] = (X[:,i] - min(X[:,i]))/(max(X[:,i]) - min(X[:,i]))
		return X, maxer, miner

	def train(self, model_signature):
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.10, random_state=42)
		print(X_train.shape)
		self.model_signature = model_signature 
		 

		if self.model_id is 3:

			if self.model_signature==1.0:
				# krnn = Sequential()
				self.krnn.add(Dense(64, input_dim=self.X.shape[1], kernel_initializer='uniform', activation='relu'))
				# self.krnn.add(Dropout(0.5))
				self.krnn.add(Dense(16, kernel_initializer='uniform', activation='relu'))
				# self.krnn.add(Dropout(0.5))
				self.krnn.add(Dense(1, kernel_initializer ='uniform', activation='sigmoid'))
			else:
				while True:	
					try:
						# print (' Tried to load file : ', self.path+'/model_krnn_%s_.h5'%(self.model_signature-1))
						self.krnn = load_model(self.path+'/model_krnn_%s_.h5'%(model_signature-1))
						#self.krnn = self.create_model(self.dropout)
               
						break
					except EnvironmentError as e:
						pass
						# print(e.errno)
						# time.sleep(1)
						# print ('ERROR in loading latest surrogate model, loading previous one in TRAIN')
				
			early_stopping = EarlyStopping(monitor='val_loss', patience=5)
			self.krnn.compile(loss='mse', optimizer='adam', metrics=['mse'])
			train_log = self.krnn.fit(X_train, y_train.ravel(), batch_size=50, epochs=20, validation_split=0.1, verbose=0, callbacks=[early_stopping])

			scores = self.krnn.evaluate(X_test, y_test.ravel(), verbose = 0)
			print("%s: %.5f" % (self.krnn.metrics_names[1], scores[1]))
		
			self.krnn.save(self.path+'/model_krnn_%s_.h5' %self.model_signature)
			print("Saved model to disk  ", self.model_signature)

			
			'''plt.plot(train_log.history["loss"], label="loss")
			plt.plot(train_log.history["val_loss"], label="val_loss") 
			plt.savefig(self.path+'/%s_0.png'%(self.model_signature))

			plt.clf()'''

			results = np.array([scores[1]])  
			print(results, 'train-metrics')


			with open(('%s/train_metrics.txt' % (self.path)),'ab') as outfile:
				np.savetxt(outfile, results)

			if self.save_surrogatedata is True:
				with open(('%s/learnsurrogate_data/X_train.csv' % (self.path)),'ab') as outfile:
					np.savetxt(outfile, X_train)
				with open(('%s/learnsurrogate_data/Y_train.csv' % (self.path)),'ab') as outfile:
					np.savetxt(outfile, y_train)	
				with open(('%s/learnsurrogate_data/X_test.csv' % (self.path)),'ab') as outfile:
					np.savetxt(outfile, X_test)
				with open(('%s/learnsurrogate_data/Y_test.csv' % (self.path)),'ab') as outfile:
					np.savetxt(outfile, y_test)
 
	def predict(self, X_load, initialized):
		 

		if self.model_id == 3:
			
			if initialized == False:
				model_sign = np.loadtxt(self.path+'/model_signature.txt')
				self.model_signature = model_sign 
				while True:
					try:
						self.krnn = load_model(self.path+'/model_krnn_%s_.h5'%self.model_signature)
						# print (' Tried to load file : ', self.path+'/model_krnn_%s_.h5'%self.model_signature)
						break
					except EnvironmentError as e:
						pass 
			
				self.krnn.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
				krnn_prediction =-1.0
				prediction = -1.0
			
			else:
				krnn_prediction = self.krnn.predict(X_load)[0]

				prediction = krnn_prediction*(self.max_Y[0,0]-self.min_Y[0,0]) + self.min_Y[0,0]

			return prediction, krnn_prediction


class ptReplica(multiprocessing.Process):

	def __init__(self, use_surrogate,  use_langevin_gradients,  learn_rate, save_surrogatedata,  w,  minlim_param, maxlim_param, samples, traindata, testdata, topology, burn_in, temperature, swap_interval, path, parameter_queue, main_process,event,surrogate_parameterqueue,surrogate_interval,surrogate_prob,surrogate_start,surrogate_resume):
		#MULTIPROCESSING VARIABLES
		multiprocessing.Process.__init__(self)
		self.processID = temperature
		self.parameter_queue = parameter_queue
		self.signal_main = main_process
		self.event =  event
		#SURROGATE VARIABLES
		self.surrogate_parameterqueue = surrogate_parameterqueue
		self.surrogate_start = surrogate_start
		self.surrogate_resume = surrogate_resume
		self.surrogate_interval = surrogate_interval
		self.surrogate_prob = surrogate_prob
		#PARALLEL TEMPERING VARIABLES
		self.temperature = temperature


		self.adapttemp =  self.temperature #* ratio  # 

		self.swap_interval = swap_interval
		self.path = path
		self.burn_in = burn_in
		#FNN CHAIN VARIABLES (MCMC)
		self.samples = samples
		self.topology = topology
		self.traindata = traindata
		self.testdata = testdata
		self.w = w

		self.minY = np.zeros((1,1))
		self.maxY = np.zeros((1,1))

		self.minlim_param = minlim_param
		self.maxlim_param = maxlim_param

		self.use_surrogate =  use_surrogate


		self.use_langevin_gradients = use_langevin_gradients



		self.save_surrogatedata =  save_surrogatedata

		self.compare_surrogate  = True
		self.sgd_depth = 1 # always should be 1 
		self.learn_rate =   learn_rate # learn rate for langevin 

		self.l_prob = 0.5  # can be evaluated for diff problems - if data too large keep this low value since the gradients cost comp time

		langevin_count = 0

 

	def rmse(self, pred, actual): 

		return np.sqrt(((pred-actual)**2).mean())

	def accuracy(self,pred,actual ):
		count = 0
		for i in range(pred.shape[0]):
			if pred[i] == actual[i]:
				count+=1 
  


		return 100*(count/pred.shape[0])

	def likelihood_func(self, fnn, data, w):
		y = data[:, self.topology[0]]
		fx, prob = fnn.evaluate_proposal(data,w)
		rmse = self.rmse(fx,y)
		z = np.zeros((data.shape[0],self.topology[2]))
		lhood = 0
		for i in range(data.shape[0]):
			for j in range(self.topology[2]):
				if j == y[i]:
					z[i,j] = 1
				lhood += z[i,j]*np.log(prob[i,j])
 

		return [lhood/self.adapttemp, fx, rmse, lhood]

	def prior_likelihood(self, sigma_squared, nu_1, nu_2, w):
		h = self.topology[1]  # number hidden neurons
		d = self.topology[0]  # number input neurons
		part1 = -1 * ((d * h + h + self.topology[2]+h*self.topology[2]) / 2) * np.log(sigma_squared)
		part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
		log_loss = part1 - part2
		return log_loss

	def run(self):
		#INITIALISING FOR FNN
		testsize = self.testdata.shape[0]
		trainsize = self.traindata.shape[0]
		samples = self.samples
		self.sgd_depth = 1
		x_test = np.linspace(0,1,num=testsize)
		x_train = np.linspace(0,1,num=trainsize)
		netw = self.topology
		y_test = self.testdata[:,netw[0]]
		y_train = self.traindata[:,netw[0]]

		w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias
		pos_w = np.ones((samples, w_size)) #Posterior for all weights
		s_pos_w = np.ones((samples, w_size)) #Surrogate Trainer
		lhood_list = np.zeros((samples,1))
		surrogate_list = np.zeros((samples ,1))
		#fxtrain_samples = np.ones((samples/100, trainsize)) #Output of regression FNN for training samples
		#fxtest_samples = np.ones((samples/100, testsize)) #Output of regression FNN for testing samples
		rmse_train  = np.zeros(samples)
		rmse_test = np.zeros(samples)
		acc_train = np.zeros(samples)
		acc_test = np.zeros(samples)
		learn_rate = 0.5

		naccept = 0
		#Random Initialisation of weights
		w = self.w
		eta = 0 #Junk variable 
		#print(w,self.temperature)
		w_proposal = np.random.randn(w_size)
		#Randomwalk Steps
		step_w = 0.025
		#Declare FNN
		fnn = Network(self.topology, self.traindata, self.testdata, learn_rate)
		#Evaluate Proposals
		pred_train, prob_train = fnn.evaluate_proposal(self.traindata,w) #	
		pred_test, prob_test = fnn.evaluate_proposal(self.testdata, w) #
		#Check Variance of Proposal
		sigma_squared = 25
		nu_1 = 0
		nu_2 = 0
		sigma_diagmat = np.zeros((w_size, w_size))  # for Equation 9 in Ref [Chandra_ICONIP2017]
		np.fill_diagonal(sigma_diagmat, step_w)

		delta_likelihood = 0.5 # an arbitrary position
		prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w)  # takes care of the gradients
		#Evaluate Likelihoods
		[likelihood, pred_train, rmsetrain, likl_without_temp] = self.likelihood_func(fnn, self.traindata, w)
		[_, pred_test, rmsetest, likl_without_temp] = self.likelihood_func(fnn, self.testdata, w)
		#Beginning Sampling using MCMC RANDOMWALK

		likelihood_copy = likelihood
		

		#accept_list = open(self.path+'/acceptlist_'+str(int(self.temperature*10))+'.txt', "a+")



		trainacc = 0
		testacc=0

		prop_list = np.zeros((samples,w_proposal.size))
		likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood
		likeh_list[0,:] = [-100, -100] # to avoid prob in calc of 5th and 95th percentile later
		surg_likeh_list = np.zeros((samples,2))
		accept_list = np.zeros(samples)

		num_accepted = 0

		is_true_lhood = True

 
		lhood_counter = 0
		lhood_counter_inf = 0 
		reject_counter = 0
		reject_counter_inf = 0

		langevin_count = 0



		pt_samples = samples * 0.6# this means that PT in canonical form with adaptive temp will work till pt  samples are reached





		burnsamples = int(self.samples * self.burn_in)

		init_count = 0


 

		surrogate_counter = 0
		
		for i in range(samples-1):

			timer1 = time.time()

			lx = np.random.uniform(0,1,1)


			ratio = ((samples -i) /(samples*1.0))

			#self.adapttemp =  self.temperature 

			if i < pt_samples:
				self.adapttemp =  self.temperature #* ratio  #  

			if i == pt_samples and init_count ==0: # move to MCMC canonical
				self.adapttemp = 1  
				[likelihood, pred_train, rmsetrain, likl_without_temp] = self.likelihood_func(fnn, self.traindata, w)
				[_, pred_test, rmsetest, likl_without_temp] = self.likelihood_func(fnn, self.testdata, w)
				init_count = 1

  
			w_proposal = np.random.normal(w, step_w, w_size)
 

   
			ku = random.uniform(0,1)   
 

			if ku<self.surrogate_prob and i>=self.surrogate_interval+1:  


				is_true_lhood = False

				if surrogate_model == None:
					minmax = np.loadtxt(self.path+'/surrogate/minmax.txt')
					self.minY[0,0] = minmax[0]
					self.maxY[0,0] = minmax[1]
					surrogate_model = surrogate("krnn",surrogate_X.copy(),surrogate_Y.copy(), self.minlim_param, self.maxlim_param, self.minY, self.maxY, self.path, self.save_surrogatedata)
					surrogate_likelihood, nn_predict = surrogate_model.predict(w_proposal.reshape(1,w_proposal.shape[0]),False)
					surrogate_likelihood = surrogate_likelihood *(1.0/self.adapttemp)
						
				elif self.surrogate_init == 0.0:
					surrogate_likelihood,  nn_predict = surrogate_model.predict(w_proposal.reshape(1,w_proposal.shape[0]), False)
					surrogate_likelihood = surrogate_likelihood *(1.0/self.adapttemp)
				else:
					surrogate_likelihood,  nn_predict = surrogate_model.predict(w_proposal.reshape(1,w_proposal.shape[0]), True)
					surrogate_likelihood = surrogate_likelihood *(1.0/self.adapttemp)

			

				likelihood_proposal = surrogate_likelihood[0]

 

				if self.compare_surrogate is True: 
					[likelihood_proposal_true, pred_train, rmsetrain, likl_without_temp] = self.likelihood_func(fnn, self.traindata, w_proposal) 
				else: 
					likelihood_proposal_true = 0
					

				print ('\nSample : ', i, ' Chain :', self.adapttemp, ' -A', likelihood_proposal_true, ' vs. P ',  likelihood_proposal, ' ---- nnPred ', nn_predict, self.minY, self.maxY )
				surrogate_counter += 1

				surg_likeh_list[i+1,0] =  likelihood_proposal_true
				surg_likeh_list[i+1,1] = likelihood_proposal
			

			else:
				is_true_lhood = True 

				surg_likeh_list[i+1,1] =  np.nan 

			  		
				  
			 	[likelihood_proposal, pred_train, rmsetrain, likl_without_temp] = self.likelihood_func(fnn, self.traindata, w_proposal)
			 	[_, pred_test, rmsetest, likl_without_temp_] = self.likelihood_func(fnn, self.testdata, w_proposal)


			 	surg_likeh_list[i+1,0] = likelihood_proposal  


			
			'''is_true_lhood = True 
			[likelihood_proposal, pred_train, rmsetrain, likl_without_temp] = self.likelihood_func(fnn, self.traindata, w_proposal)
			[_, pred_test, rmsetest, likl_without_temp_] = self.likelihood_func(fnn, self.testdata, w_proposal) '''


			 

			prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal)  # takes care of the gradients
			
			diff_likelihood = likelihood_proposal -   likelihood_copy # (lhood_list[i,]  /self.adapttemp)  #

			diff_prior = prior_prop - prior_current
			try:
				mh_prob = min(1, math.exp(diff_likelihood  + diff_prior))
			except OverflowError as e:
				mh_prob = 1



			accept_list[i+1] = naccept 


			#likeh_list[i+1,0] = surrogate_var
			#prop_list[i+1,] = v_proposal


			u = random.uniform(0, 1)
			
			prop_list[i+1,] = w_proposal	
			likeh_list[i+1,0] = likl_without_temp


			if u < mh_prob:
				naccept  =  naccept + 1
				likelihood = likelihood_proposal
				likelihood_copy = likelihood_proposal
				prior_current = prior_prop
				w = w_proposal
					 

				pos_w[i + 1,] = w_proposal 

				if is_true_lhood is  True:
					lhood_list[i+1,] = (likelihood*self.adapttemp) 

					#fxtrain_samples[i + 1,] = pred_train
					#fxtest_samples[i + 1,] = pred_test
					rmse_train[i + 1,] = rmsetrain
					rmse_test[i + 1,] = rmsetest 
					acc_train[i+1,] = self.accuracy(pred_train, y_train )   
					acc_test[i+1,] = self.accuracy(pred_test, y_test )


					lhood_counter = lhood_counter + 1

					print (i, self.adapttemp, lhood_counter ,   likelihood,   mh_prob, math.exp(diff_likelihood  + diff_prior),  diff_likelihood ,  diff_prior, acc_train[i+1,], acc_test[i+1,], self.adapttemp, 'accepted')

 

				else:
					lhood_list[i+1,] = np.inf


					#fxtrain_samples[i + 1,] = np.inf
					#fxtest_samples[i + 1,] = np.inf
					rmse_train[i + 1,] = np.inf
					rmse_test[i + 1,] = np.inf
					acc_train[i+1,] = np.inf
					acc_test[i+1,] = np.inf 

					'''rmse_train[i + 1,] =   rmse_train[lhood_counter,]
					rmse_test[i + 1,] =  rmse_test[lhood_counter,]
					acc_train[i+1,] =  acc_train[lhood_counter,]
					acc_test[i+1,] =  acc_test[lhood_counter,] '''


					lhood_counter_inf = lhood_counter_inf + 1

					#print (i,lhood_counter ,   likelihood, self.adapttemp,   acc_train[i+1,], acc_test[i+1,],  'accepted sur')
					#print (i,lhood_counter ,   likelihood,   mh_prob, math.exp(diff_likelihood  + diff_prior),  diff_likelihood ,  diff_prior, acc_train[i+1,], acc_test[i+1,], self.adapttemp, '  not accepted')

			

			else:       
				pos_w[i+1,] = pos_w[i,] 

				if is_true_lhood is True:
					lhood_list[i+1,] = (likelihood_proposal*self.adapttemp)
				
					#fxtrain_samples[i + 1,] = fxtrain_samples[i,]
					#fxtest_samples[i + 1,] = fxtest_samples[i,]
					rmse_train[i + 1,] =   rmse_train[i,]
					rmse_test[i + 1,] =  rmse_test[i,]
					acc_train[i+1,] =  acc_train[i,]
					acc_test[i+1,] =  acc_test[i,] 

					reject_counter = reject_counter + 1



					#print (i,lhood_counter ,   likelihood,   acc_train[lhood_counter,], acc_test[lhood_counter,],  self.adapttemp, 'rejected  true-lhood ')
				else:
					lhood_list[i+1,] = np.inf 


					#fxtrain_samples[i + 1,] = np.inf
					#fxtest_samples[i + 1,] = np.inf
					rmse_train[i + 1,] = np.inf
					rmse_test[i + 1,] = np.inf
					acc_train[i+1,] = np.inf
					acc_test[i+1,] = np.inf 
					'''rmse_train[i + 1,] =   rmse_train[lhood_counter,]
					rmse_test[i + 1,] =  rmse_test[lhood_counter,]
					acc_train[i+1,] =  acc_train[lhood_counter,]
					acc_test[i+1,] =  acc_test[lhood_counter,] '''


					reject_counter_inf = reject_counter_inf + 1


					#print (i,lhood_counter ,   likelihood, self.adapttemp, rmsetrain, rmsetest, acc_train[i+1,], acc_test[i+1,],  'accepted surr ')



				
			#SWAPPING PREP
			if i%self.swap_interval == 0:
				param = np.concatenate([w, np.asarray([eta]).reshape(1), np.asarray([likelihood]),np.asarray([self.adapttemp]),np.asarray([i])])
				self.parameter_queue.put(param)
				self.signal_main.set()
				self.event.wait()
				# retrieve parameters fom queues if it has been swapped
				if not self.parameter_queue.empty() : 
					try:
						result =  self.parameter_queue.get()
						w= result[0:w.size]     
						eta = result[w.size]
						likelihood = result[w.size+1]
					except:
						print ('error') 

			if (i%self.surrogate_interval == 0) and (i!=0):
				#Train the surrogate with the posteriors and likelihood
				surrogate_X, surrogate_Y = prop_list[i+1-self.surrogate_interval:i,:],likeh_list[i+1-self.surrogate_interval:i,0]

				surrogate_Y = surrogate_Y.reshape(surrogate_Y.shape[0],1)
				param = np.concatenate([surrogate_X, surrogate_Y],axis=1)
 
				self.surrogate_parameterqueue.put(param)
				self.surrogate_start.set()
				self.surrogate_resume.wait() 
				
				model_sign = np.loadtxt(self.path+'/surrogate/model_signature.txt')
				self.model_signature = model_sign 
				

				if self.model_signature==1.0:
					minmax = np.loadtxt(self.path+'/surrogate/minmax.txt')
					self.minY[0,0] = minmax[0]
					self.maxY[0,0] = minmax[1]
					# print 'min ', self.minY, ' max ', self.maxY
					dummy_X = np.zeros((1,1))
					dummy_Y = np.zeros((1,1))
					surrogate_model = surrogate("krnn", dummy_X, dummy_Y, self.minlim_param, self.maxlim_param, self.minY, self.maxY, self.path, self.save_surrogatedata )
				 
				self.surrogate_init,  nn_predict  = surrogate_model.predict(w_proposal.reshape(1,w_proposal.shape[0]), False)				
				print("Surrogate init ", self.surrogate_init , " - should be -1") 



		param = np.concatenate([w, np.asarray([eta]).reshape(1), np.asarray([likelihood]),np.asarray([self.adapttemp]),np.asarray([i])])
		#print('SWAPPED PARAM',self.adapttemp,param)
		self.parameter_queue.put(param)
		param = np.concatenate([s_pos_w[i-self.surrogate_interval:i,:],lhood_list[i-self.surrogate_interval:i,:]],axis=1)
		self.surrogate_parameterqueue.put(param) 

		print ((naccept*100 / (samples * 1.0)), '% was accepted')


		print (surrogate_counter/samples * 100 , ' % use of surogate')

		accept_ratio = naccept / (samples * 1.0) * 100




		print(lhood_counter, lhood_counter_inf, reject_counter, reject_counter_inf, '   lhood_counter, lhood_counter_inf, reject_counter, reject_counter_inf')
  

		file_name = self.path+'/posterior/pos_w/'+'chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,pos_w ) 
		'''file_name = self.path+'/predictions/fxtrain_samples_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, fxtrain_samples, fmt='%1.2f')
		file_name = self.path+'/predictions/fxtest_samples_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, fxtest_samples, fmt='%1.2f')		'''
		file_name = self.path+'/predictions/rmse_test_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, rmse_test, fmt='%1.2f')		
		file_name = self.path+'/predictions/rmse_train_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, rmse_train, fmt='%1.2f')


		file_name = self.path+'/predictions/acc_test_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, acc_test, fmt='%1.2f')		
		file_name = self.path+'/predictions/acc_train_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, acc_train, fmt='%1.2f') 

 
		file_name = self.path+'/posterior/surg_likelihood/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,surg_likeh_list, fmt='%1.4f') 

		file_name = self.path+'/posterior/pos_likelihood/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,likeh_list, fmt='%1.4f') 

		
		file_name = self.path + '/posterior/accept_list/chain_' + str(self.temperature) + '_accept.txt'
		np.savetxt(file_name, [accept_ratio], fmt='%1.4f')

		file_name = self.path + '/posterior/accept_list/chain_' + str(self.temperature) + '.txt'
		np.savetxt(file_name, accept_list, fmt='%1.4f')

		 


 



		self.signal_main.set()
		self.surrogate_start.set()

class ParallelTempering:

	def __init__(self, use_surrogate,  use_langevin_gradients, learn_rate,  save_surrogatedata, traindata, testdata, topology, num_chains, maxtemp, NumSample, swap_interval, surrogate_interval, surrogate_prob, path, path_db):
		#FNN Chain variables
		self.traindata = traindata
		self.testdata = testdata
		self.topology = topology
		self.num_param = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
		#Parallel Tempering variables
		self.swap_interval = swap_interval
		self.path = path
		self.path_db = path_db
		self.maxtemp = maxtemp
		self.num_swap = 0
		self.total_swap_proposals = 0
		self.num_chains = num_chains
		self.chains = []
		self.temperatures = []
		self.NumSamples = int(NumSample/self.num_chains)
		self.sub_sample_size = max(1, int( 0.05* self.NumSamples))
		# create queues for transfer of parameters between process chain
		self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
		self.chain_queue = multiprocessing.JoinableQueue()	
		self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]
		self.event = [multiprocessing.Event() for i in range (self.num_chains)]
		# create variables for surrogates
		self.surrogate_interval = surrogate_interval
		self.surrogate_prob = surrogate_prob
		self.surrogate_resume_events = [multiprocessing.Event() for i in range(self.num_chains)]
		self.surrogate_start_events = [multiprocessing.Event() for i in range(self.num_chains)]
		self.surrogate_parameterqueues = [multiprocessing.Queue() for i in range(self.num_chains)]
		self.surrchain_queue = multiprocessing.JoinableQueue()
		self.all_param = None
		self.geometric = True # True (geometric)  False (Linear)

		self.minlim_param = 0.0
		self.maxlim_param = 0.0
		self.minY = np.zeros((1,1))
		self.maxY = np.ones((1,1))

		self.model_signature = 0.0

		self.use_surrogate = use_surrogate


		self.save_surrogatedata =  save_surrogatedata

		self.use_langevin_gradients =  use_langevin_gradients

		self.learn_rate = learn_rate

	def default_beta_ladder(self, ndim, ntemps, Tmax): #https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
		"""
		Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
		arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
		Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
		this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
		<http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
		``ntemps`` is also specified. 
		"""

		if type(ndim) != int or ndim < 1:
			raise ValueError('Invalid number of dimensions specified.')
		if ntemps is None and Tmax is None:
			raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
		if Tmax is not None and Tmax <= 1:
			raise ValueError('``Tmax`` must be greater than 1.')
		if ntemps is not None and (type(ntemps) != int or ntemps < 1):
			raise ValueError('Invalid number of temperatures specified.')

		tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
						  2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
						  2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
						  1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
						  1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
						  1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
						  1.51901, 1.50881, 1.49916, 1.49, 1.4813,
						  1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
						  1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
						  1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
						  1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
						  1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
						  1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
						  1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
						  1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
						  1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
						  1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
						  1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
						  1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
						  1.26579, 1.26424, 1.26271, 1.26121,
						  1.25973])

		if ndim > tstep.shape[0]:
			# An approximation to the temperature step at large
			# dimension
			tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
		else:
			tstep = tstep[ndim-1]

		appendInf = False
		if Tmax == np.inf:
			appendInf = True
			Tmax = None
			ntemps = ntemps - 1

		if ntemps is not None:
			if Tmax is None:
				# Determine Tmax from ntemps.
				Tmax = tstep ** (ntemps - 1)
		else:
			if Tmax is None:
				raise ValueError('Must specify at least one of ``ntemps'' and '
								 'finite ``Tmax``.')

			# Determine ntemps from Tmax.
			ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

		betas = np.logspace(0, -np.log10(Tmax), ntemps)
		if appendInf:
			# Use a geometric spacing, but replace the top-most temperature with
			# infinity.
			betas = np.concatenate((betas, [0]))

		return betas
		
	def assign_temperatures(self):
		# #Linear Spacing
		# temp = 2
		# for i in range(0,self.num_chains):
		# 	self.temperatures.append(temp)
		# 	temp += 2.5 #(self.maxtemp/self.num_chains)
		# 	print (self.temperatures[i])
		#Geometric Spacing

		if self.geometric == True:
			betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)      
			for i in range(0, self.num_chains):         
				self.temperatures.append(np.inf if betas[i] is 0 else 1.0/betas[i])
				print (self.temperatures[i])
		else:
 
			tmpr_rate = (self.maxtemp /self.num_chains)
			temp = 1
			for i in xrange(0, self.num_chains):            
				self.temperatures.append(temp)
				temp += tmpr_rate
				print(self.temperatures[i])


	def initialize_chains(self,  burn_in):
		self.burn_in = burn_in
		self.assign_temperatures()
		self.minlim_param = np.repeat([-100] , self.num_param)  # priors for nn weights
		self.maxlim_param = np.repeat([100] , self.num_param)
 

		w = np.random.randn(self.num_param)
		
		for i in range(0, self.num_chains):
			self.chains.append(ptReplica(self.use_surrogate,  self.use_langevin_gradients, self.learn_rate, self.save_surrogatedata, w,  self.minlim_param, self.maxlim_param, self.NumSamples,self.traindata,self.testdata,self.topology,self.burn_in,self.temperatures[i],self.swap_interval,self.path,self.parameter_queue[i],self.wait_chain[i],self.event[i],self.surrogate_parameterqueues[i],self.surrogate_interval,self.surrogate_prob,self.surrogate_start_events[i],self.surrogate_resume_events[i]))

	def surr_procedure(self,queue):

		if queue.empty() is False:
			return queue.get()
		else:
			return
	
	def swap_procedure(self, parameter_queue_1, parameter_queue_2):
		if parameter_queue_2.empty() is False and parameter_queue_1.empty() is False:
			param1 = parameter_queue_1.get()
			param2 = parameter_queue_2.get()
			w1 = param1[0:self.num_param]
			eta1 = param1[self.num_param]
			lhood1 = param1[self.num_param+1]
			T1 = param1[self.num_param+2]
			w2 = param2[0:self.num_param]
			eta2 = param2[self.num_param]
			lhood2 = param2[self.num_param+1]
			T2 = param2[self.num_param+2]
			#print('yo')
			#SWAPPING PROBABILITIES
			try:
				swap_proposal =  min(1,0.5*np.exp(min(709, lhood2 - lhood1)))
			except OverflowError:
				swap_proposal = 1
			u = np.random.uniform(0,1)
			if u < swap_proposal: 
				self.total_swap_proposals += 1
				self.num_swap += 1
				param_temp =  param1
				param1 = param2
				param2 = param_temp
			return param1, param2
		else:
			self.total_swap_proposals += 1
			return

	def surrogate_trainer(self,params):
		#X = params[:,:self.num_param]
		#Y = params[:,self.num_param].reshape(X.shape[0],1)
		#indices = np.where(Y==np.inf)[0]
		#X = np.delete(X, indices, axis=0)
		#Y = np.delete(Y,indices, axis=0)
		#surrogate_model = surrogate("nn",X,Y,self.path)
		#surrogate_model.train()


		X = params[:,:self.num_param]
		Y = params[:,self.num_param].reshape(X.shape[0],1)
		
		for i in range(Y.shape[1]):
			min_Y = min(Y[:,i])
			max_Y = max(Y[:,i])
			self.minY[0,i] =   min_Y * 3
			self.maxY[0,i] = -1#max_Y 

		self.model_signature += 1.0 
		if self.model_signature == 1.0:
			np.savetxt(self.path+'/surrogate/minmax.txt',[self.minY[0, 0], self.maxY[0, 0]])
		
		np.savetxt(self.path+'/surrogate/model_signature.txt', [self.model_signature])
		
		Y= self.normalize_likelihood(Y)
		indices = np.where(Y==np.inf)[0]
		X = np.delete(X, indices, axis=0)
		Y = np.delete(Y,indices, axis=0)
		surrogate_model = surrogate("krnn", X , Y , self.minlim_param, self.maxlim_param, self.minY, self.maxY, self.path, self.save_surrogatedata )
		surrogate_model.train(self.model_signature)


	def normalize_likelihood(self, Y):
		for i in range(Y.shape[1]):
			if self.model_signature == 1.0:
				min_Y = min(Y[:,i])
				max_Y = max(Y[:,i])
				# self.minY[0,i] = 1 #For Tau Squared
				# self.maxY[0,i] = max_Y 
				

				# min -115 and max -96
				self.maxY[0,i] = -1 #max_Y
				self.minY[0,i] =  min_Y * 2

			# Y[:,i] = ([:,i] - min_Y)/(max_Y - min_Y)
			
			Y[:,i] = (Y[:,i] - self.minY[0,0])/(self.maxY[0,0]-self.minY[0,0])

		return Y	
		
	def plot_figure(self, list, title): 

		list_points =  list

		fname = self.path
		width = 9 

		font = 9

		fig = plt.figure(figsize=(10, 12))
		ax = fig.add_subplot(111)
 

		slen = np.arange(0,len(list),1) 
		 
		fig = plt.figure(figsize=(10,12))
		ax = fig.add_subplot(111)
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
		ax.set_title(' Posterior distribution', fontsize=  font+2)#, y=1.02)
	
		ax1 = fig.add_subplot(211) 

		n, rainbins, patches = ax1.hist(list_points,  bins = 20,  alpha=0.5, facecolor='sandybrown', normed=False)	
 
  
		color = ['blue','red', 'pink', 'green', 'purple', 'cyan', 'orange','olive', 'brown', 'black']

		ax1.grid(True)
		ax1.set_ylabel('Frequency',size= font+1)
		ax1.set_xlabel('Parameter values', size= font+1)
	
		ax2 = fig.add_subplot(212)

		list_points = np.asarray(np.split(list_points,  self.num_chains ))
 

 

		ax2.set_facecolor('#f2f2f3') 
		ax2.plot( list_points.T , label=None)
		ax2.set_title(r'Trace plot',size= font+2)
		ax2.set_xlabel('Samples',size= font+1)
		ax2.set_ylabel('Parameter values', size= font+1) 

		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		 
 
		plt.savefig(fname + '/' + title  + '_pos_.png', bbox_inches='tight', dpi=300, transparent=False)
		plt.clf()
 
	def run_chains(self): 
		# only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
		swap_proposal = np.ones(self.num_chains-1) 
		# create parameter holders for paramaters that will be swapped
		replica_param = np.zeros((self.num_chains, self.num_param))  
		lhood = np.zeros(self.num_chains)
		# Define the starting and ending of MCMC Chains
		start = 0
		end = self.NumSamples-1
		number_exchange = np.zeros(self.num_chains)
		filen = open(self.path + '/num_exchange.txt', 'a')
		#RUN MCMC CHAINS
		for l in range(0,self.num_chains):
			self.chains[l].start_chain = start
			self.chains[l].end = end
		for j in range(0,self.num_chains):        
			self.chains[j].start()
		#SWAP PROCEDURE
		
		while True:
			for k in range(0,self.num_chains):
				self.wait_chain[j].wait()
				#print(chain_num)
			for k in range(0,self.num_chains-1):
				#print('starting swap')
				self.chain_queue.put(self.swap_procedure(self.parameter_queue[k],self.parameter_queue[k+1])) 
				while True:
					if self.chain_queue.empty():
						self.chain_queue.task_done()
						#print(k,'EMPTY QUEUE')
						break
					swap_process = self.chain_queue.get()
					#print(swap_process)
					if swap_process is None:
						self.chain_queue.task_done()
						#print(k,'No Process')
						break
					param1, param2 = swap_process
					
					self.parameter_queue[k].put(param1)
					self.parameter_queue[k+1].put(param2)
			for k in range (self.num_chains):
					#print(k)
					self.event[k].set()
			count = 0
			# Surrogate's Events:

 
			
			for k in range(0,self.num_chains):
				self.surrogate_start_events[k].wait()
			for k in range(0,self.num_chains):
				self.surrchain_queue.put(self.surr_procedure(self.surrogate_parameterqueues[k]))
				params = None
				while True:
					if self.surrchain_queue.empty():
						self.surrchain_queue.task_done()
						#print(k,'EMPTY QUEUE')
						break
					params = self.surrchain_queue.get()
					#print(params, 'param in surr')

					if params is not None:
						self.surrchain_queue.task_done()
						#print(k,'No Process')
						break
				if params is not None:
					 all_param = np.asarray(params if not ('all_param' in locals()) else np.concatenate([ all_param,params],axis=0)) #could be bug

					#print(self.all_param, ' all_param')


			if ('all_param' in locals()):
				if all_param.shape == (self.num_chains*(self.surrogate_interval-1),self.num_param+1):
					self.surrogate_trainer( all_param)
					del  all_param
					for k in range(self.num_chains):
						self.surrogate_resume_events[k].set()
			
			######################
			for i in range(self.num_chains):
				if self.chains[i].is_alive() is False:
					count+=1
			#print(count)
			if count == self.num_chains  :
				print(count)
				break
			
		
		#JOIN THEM TO MAIN PROCESS
		for j in range(0,self.num_chains):
			self.chains[j].join()
		self.chain_queue.join()
		for j in range(0,self.num_chains):
			self.parameter_queue[i].close()
			self.parameter_queue[i].join_thread()
			self.surrogate_parameterqueues[i].close()
			self.surrogate_parameterqueues[i].join_thread()



		time.sleep(5)
		 

		pos_w, fx_train, fx_test,   rmse_train, rmse_test, acc_train, acc_test,  likelihood_vec , accept_list,  rmse_surr, surr_list, accept   = self.show_results()







		# for s in range(self.num_param):  
		# 	self.plot_figure(pos_w[s,:], 'pos_distri_'+str(s)) 
		#print("accuracies", max(acc_train), max(acc_test))
		print("NUMBER OF SWAPS =", self.num_swap)
		swap_perc = self.num_swap*100/self.total_swap_proposals 
		#return (pos_w, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_list)

		return pos_w, fx_train, fx_test,  rmse_train, rmse_test, acc_train, acc_test,  accept_list, swap_perc,  likelihood_vec, rmse_surr, surr_list, accept



	def show_results(self):

		burnin = int(self.NumSamples*self.burn_in)
 
		likelihood_rep = np.zeros((self.num_chains, self.NumSamples  -1, 2)) # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
		surg_likelihood = np.zeros((self.num_chains, self.NumSamples -1 , 2)) # index 1 for likelihood proposal and for gp_prediction
		accept_percent = np.zeros((self.num_chains, 1))
		accept_list = np.zeros((self.num_chains, self.NumSamples )) 
 
		pos_w = np.zeros((self.num_chains,self.NumSamples - burnin, self.num_param)) 

		fx_train_all  = np.zeros((self.num_chains,self.NumSamples - burnin, self.traindata.shape[0]))
		rmse_train = np.zeros((self.num_chains,self.NumSamples - burnin))
		acc_train = np.zeros((self.num_chains,self.NumSamples - burnin))
		fx_test_all  = np.zeros((self.num_chains,self.NumSamples - burnin, self.testdata.shape[0]))
		rmse_test = np.zeros((self.num_chains,self.NumSamples - burnin))
		acc_test = np.zeros((self.num_chains,self.NumSamples - burnin))
 
		
		 
		for i in range(self.num_chains):
			file_name = self.path+'/posterior/pos_w/'+'chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			pos_w[i,:,:] = dat[burnin:,:]  

			file_name = self.path + '/posterior/pos_likelihood/'+'chain_' + str(self.temperatures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			likelihood_rep[i, :] = dat[1:,:] 

			file_name = self.path + '/posterior/surg_likelihood/'+'chain_' + str(self.temperatures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			surg_likelihood[i, :] = dat[1:,:] 

			file_name = self.path + '/posterior/accept_list/' + 'chain_'  + str(self.temperatures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			accept_list[i, :] = dat 

			file_name = self.path + '/posterior/accept_list/' + 'chain_' + str(self.temperatures[i]) + '_accept.txt'
			dat = np.loadtxt(file_name) 
			accept_percent[i, :] = dat  
 
			'''file_name = self.path+'/predictions/fxtrain_samples_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			fx_train_all[i,:,:] = dat[burnin:,:]

			file_name = self.path+'/predictions/fxtest_samples_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			fx_test_all[i,:,:] = dat[burnin:,:]'''	

			file_name = self.path+'/predictions/rmse_test_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			rmse_test[i,:] = dat[burnin:]	

			file_name = self.path+'/predictions/rmse_train_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			rmse_train[i,:] = dat[burnin:]

			file_name = self.path+'/predictions/acc_test_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			acc_test[i,:] = dat[burnin:]	

			file_name = self.path+'/predictions/acc_train_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			acc_train[i,:] = dat[burnin:]


		posterior = pos_w.transpose(2,0,1).reshape(self.num_param,-1)  

		fx_train = fx_train_all.transpose(2,0,1).reshape(self.traindata.shape[0],-1)  # need to comment this if need to save memory 
		fx_test = fx_test_all.transpose(2,0,1).reshape(self.testdata.shape[0],-1) 

		#fx_test = fxtest_samples.reshape(self.num_chains*(self.NumSamples - burnin), self.testdata.shape[0]) # konarks version
 

		likelihood_vec = likelihood_rep.transpose(2,0,1).reshape(2,-1) 
		surg_likelihood_vec = surg_likelihood.transpose(2,0,1).reshape(2,-1)

		rmse_train = rmse_train.reshape(self.num_chains*(self.NumSamples - burnin), 1)
		acc_train = acc_train.reshape(self.num_chains*(self.NumSamples - burnin), 1)
		rmse_test = rmse_test.reshape(self.num_chains*(self.NumSamples - burnin), 1)
		acc_test = acc_test.reshape(self.num_chains*(self.NumSamples - burnin), 1)

		rmse_surr =0

		surr_list = []

 

		if self.use_surrogate is True: 

			surr_list = surg_likelihood_vec.T

			surrogate_likl = surg_likelihood_vec.T
			surrogate_likl = surrogate_likl[~np.isnan(surrogate_likl).any(axis=1)]


			rmse_surr =  np.sqrt(((surrogate_likl[:,1]-surrogate_likl[:,0])**2).mean())


			print(rmse_surr, ' rmse_surr')

			print(surrogate_likl[:,0], 'Model Likelihood')

			print(surrogate_likl[:,1], 'Surrogate Likelihood')


			slen = np.arange(0,surrogate_likl.shape[0],1)
			fig = plt.figure(figsize = (12,12))
			ax = fig.add_subplot(111)
			ax.set_facecolor('#f2f2f3')
			surrogate_plot = ax.plot(slen,surrogate_likl[:,1],linestyle='-', linewidth= 1, color= 'b', label= 'Surrogate Likelihood')
			model_plot = ax.plot(slen,surrogate_likl[:,0],linestyle= '--', linewidth = 1, color = 'k', label = 'True Likelihood')
			ax.set_xlabel('Combined samples for all replica ',size= 16)
			ax.set_ylabel(' Likelihood', size= 16)
			ax.set_xlim([0,np.amax(slen)])
			#ax.legend((surrogate_plot, model_plot),('Surrogate', 'True'))

			ax.legend(loc='best')
			fig.tight_layout()
			fig.subplots_adjust(top=0.88)
			plt.savefig('%s/surrogate_likl.png'% (self.path), dpi=300, transparent=False)

			plt.savefig('%s/surrogate_likl.png'% (self.path_db), dpi=300, transparent=False)
			plt.clf() 


			# creating test data set for testing with SGD-FNN verification
			y_norm = likelihood_vec.T[:,0]
			y_norm = y_norm.reshape(y_norm.shape[0],1)
			Y_norm = self.normalize_likelihood(y_norm)

		


			np.savetxt(self.path + '/surrogate/surg_likelihood.txt', surrogate_likl, fmt='%1.5f')



		accept = np.sum(accept_percent)/self.num_chains 

		np.savetxt(self.path + '/pos_param.txt', posterior.T) 
		
		np.savetxt(self.path + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')

		np.savetxt(self.path + '/acc_train.txt', acc_train, fmt='%1.2f')
  
		np.savetxt(self.path + '/acc_test.txt', acc_test, fmt='%1.2f')
 

		return posterior, fx_train_all, fx_test_all,   rmse_train, rmse_test,  acc_train, acc_test,  likelihood_vec.T, accept_list,  rmse_surr, surr_list, accept 

	def make_directory (self, directory): 
		if not os.path.exists(directory):
			os.makedirs(directory)

def main():


	if(len(sys.argv)!=6):
		sys.exit('not right input format. give problem num [1 - 8] ')

 

	problem = int(sys.argv[1])  # get input

	Samples = int(sys.argv[2]) 

	surrogate_prob = float(sys.argv[3])

	surrogate_intervalratio = float(sys.argv[4])

	print(problem, ' problem')


  
 

#problem ={ i
	separate_flag = False
	print(problem, ' problem')

	#DATA PREPROCESSING 
	if problem == 1: #Wine Quality White
		data  = np.genfromtxt('DATA/winequality-red.csv',delimiter=';')
		data = data[1:,:] #remove Labels
		classes = data[:,11].reshape(data.shape[0],1)
		features = data[:,0:11]
		separate_flag = True
		name = "winequality-red"
		hidden = 50
		ip = 11 #input
		output = 10
		#NumSample = 50000 
	if problem == 3: #IRIS
		data  = np.genfromtxt('DATA/iris.csv',delimiter=';')
		classes = data[:,4].reshape(data.shape[0],1)-1
		features = data[:,0:4]

		separate_flag = True
		name = "iris"
		hidden = 8  #12
		ip = 4 #input
		output = 3
		#NumSample = 50000
	if problem == 2: #Wine Quality White
		data  = np.genfromtxt('DATA/winequality-white.csv',delimiter=';')
		data = data[1:,:] #remove Labels
		classes = data[:,11].reshape(data.shape[0],1)
		features = data[:,0:11]
		separate_flag = True
		name = "winequality-white"
		hidden = 50
		ip = 11 #input
		output = 10
		#NumSample = 50000 
	if problem == 4: #Ionosphere
		traindata = np.genfromtxt('DATA/Ions/Ions/ftrain.csv',delimiter=',')[:,:-1]
		testdata = np.genfromtxt('DATA/Ions/Ions/ftest.csv',delimiter=',')[:,:-1]
		name = "Ionosphere"
		hidden = 15 #50
		ip = 34 #input
		output = 2
		#NumSample = 50000 
	if problem == 5: #Cancer
		traindata = np.genfromtxt('DATA/Cancer/ftrain.txt',delimiter=' ')[:,:-1]
		testdata = np.genfromtxt('DATA/Cancer/ftest.txt',delimiter=' ')[:,:-1]
		name = "Cancer"
		hidden = 8 # 12
		ip = 9 #input
		output = 2
		#NumSample =  50000 

		print(' cancer')

	if problem == 6: #Bank additional
		data = np.genfromtxt('DATA/Bank/bank-processed.csv',delimiter=';')
		classes = data[:,20].reshape(data.shape[0],1)
		features = data[:,0:20]
		separate_flag = True
		name = "bank-additional"
		hidden = 50
		ip = 20 #input
		output = 2
		#NumSample = 50000 
	if problem == 7: #PenDigit
		traindata = np.genfromtxt('DATA/PenDigit/train.csv',delimiter=',')
		testdata = np.genfromtxt('DATA/PenDigit/test.csv',delimiter=',')
		name = "PenDigit"
		for k in range(16):
			mean_train = np.mean(traindata[:,k])
			dev_train = np.std(traindata[:,k]) 
			traindata[:,k] = (traindata[:,k]-mean_train)/dev_train
			mean_test = np.mean(testdata[:,k])
			dev_test = np.std(testdata[:,k]) 
			testdata[:,k] = (testdata[:,k]-mean_test)/dev_test
		ip = 16
		hidden = 30
		output = 10

		#NumSample = 50000
	if problem == 8: #Chess
		data  = np.genfromtxt('DATA/chess.csv',delimiter=';')
		classes = data[:,6].reshape(data.shape[0],1)
		features = data[:,0:6]
		separate_flag = True
		name = "chess"
		hidden = 25
		ip = 6 #input
		output = 18

		#NumSample = 50000 


			# Rohits set of problems - processed data




	#Separating data to train and test
	if separate_flag is True:
		#Normalizing Data
		for k in range(ip):
			mean = np.mean(features[:,k])
			dev = np.std(features[:,k])
			features[:,k] = (features[:,k]-mean)/dev
		train_ratio = 0.6 #Choosable
		indices = np.random.permutation(features.shape[0])
		traindata = np.hstack([features[indices[:np.int(train_ratio*features.shape[0])],:],classes[indices[:np.int(train_ratio*features.shape[0])],:]])
		testdata = np.hstack([features[indices[np.int(train_ratio*features.shape[0])]:,:],classes[indices[np.int(train_ratio*features.shape[0])]:,:]])


	topology = [ip, hidden, output]

	print(topology, ' topology')

	netw = topology


	y_test =  testdata[:,netw[0]]
	y_train =  traindata[:,netw[0]]

	NumSample = Samples


	maxtemp = 4 
	num_chains = 10
	swap_interval = 100000  #  #how ofen you swap neighbours
	burn_in = 0.6
	surrogate_interval = int(surrogate_intervalratio * (NumSample/num_chains))

	#surrogate_prob = 0.5 
	use_surrogate = True # if you set this to false, you get canonical PT - also make surrogate prob 0


	foldername = sys.argv[5] 

	problemfolder = '/home/rohit/Desktop/SurrogatePT/'+foldername  # change this to your directory for results output - produces large datasets

	problemfolder_db = foldername  # save main results





	filename = ""
	run_nb = 0
	while os.path.exists( problemfolder+name+'_%s' % (run_nb)):
		run_nb += 1
	if not os.path.exists( problemfolder+name+'_%s' % (run_nb)):
		os.makedirs(  problemfolder+name+'_%s' % (run_nb)) 
		path = (problemfolder+ name+'_%s' % (run_nb))

	filename = ""
	run_nb = 0
	while os.path.exists( problemfolder_db+name+'_%s' % (run_nb)):
		run_nb += 1
	if not os.path.exists( problemfolder_db+name+'_%s' % (run_nb)):
		os.makedirs(  problemfolder_db+name+'_%s' % (run_nb))
		path_db = (problemfolder_db+ name+'_%s' % (run_nb)) 



	#make_directory('SydneyResults')
	resultingfile = open( path+'/master_result_file.txt','a+')


	resultingfile_db = open( path_db+'/master_result_file.txt','a+') 

	#expfile = open( path+'/expdesign_file.txt','w')

	
	save_surrogatedata = False # just to save surrogate data for analysis - set false since it will gen lots data

	use_langevin_gradients = False

	learn_rate = 0.01

	pt_samples = int(0.6 * NumSample/num_chains)   # this is for PT first stage. then sampling becomes MCMC canonical later

	timer = time.time()
	#path = "SydneyResults/"+name+"_results_"+str(NumSample)+"_"+str(maxtemp)+"_"+str(num_chains)+"_"+str(swap_ratio)+"_"+str(surrogate_interval)+"_"+str(surrogate_prob)
	

	pt = ParallelTempering(use_surrogate,  use_langevin_gradients, learn_rate,  save_surrogatedata, traindata, testdata, topology, num_chains, maxtemp, NumSample, swap_interval, surrogate_interval, surrogate_prob, path, path_db)

	directories = [  path+'/predictions/', path+'/posterior', path+'/results', path+'/surrogate', path+'/surrogate/learnsurrogate_data', path+'/posterior/pos_w',  path+'/posterior/pos_likelihood',path+'/posterior/surg_likelihood',path+'/posterior/accept_list'  ]

	for d in directories:
		pt.make_directory((filename)+ d)	



	pt.initialize_chains(  burn_in)

	#pos_w, fx_train, fx_test,   rmse_train, rmse_test, accept_total,  likelihood_rep   
	(pos_w, fx_train, fx_test,  rmse_train, rmse_test, acc_train, acc_test, accept_list, swap_perc,  likelihood_rep, rmse_surr, surr_list, accept  ) = pt.run_chains()

   
 
	timer2 = time.time()

	timetotal = (timer2 - timer) /60
	print ((timetotal), 'min taken')
 


	x_index = np.where(acc_train==np.inf)  
	acc_train = np.delete(acc_train, x_index, axis = 0) 
	acc_tr = np.mean(acc_train [:])   
	acctr_std = np.std(acc_train[:]) 
	acctr_max = np.amax(acc_train[:])

	x_index = np.where(acc_test==np.inf)  
	acc_test = np.delete(acc_test, x_index, axis = 0) 
	acc_tes = np.mean(acc_test[:])
	acctest_std = np.std(acc_test[:]) 
	acctes_max = np.amax(acc_test[:])



	x_index = np.where(rmse_train==np.inf)  
	rmse_train = np.delete(rmse_train, x_index, axis = 0) 
	rmse_tr = np.mean(rmse_train[:])
	rmsetr_std = np.std(rmse_train[:])
	rmsetr_max = np.amax(acc_train[:])

	x_index = np.where(rmse_test==np.inf)  
	rmse_test = np.delete(rmse_test, x_index, axis = 0) 
	rmse_tes = np.mean(rmse_test[:])
	rmsetest_std = np.std(rmse_test[:])
	rmsetes_max = np.amax(acc_train[:])


	surrgate_intervalres = np.loadtxt(path+'/surrogate/train_metrics.txt') 
	print(surrgate_intervalres, ' surrgate_intervalres') 

	fig = plt.figure()
	ax = fig.add_subplot(111)
 


	x = np.arange(0, surrgate_intervalres.shape[0], 1)
	x = x.astype(int)
	ax.bar(x, surrgate_intervalres  ) 
	ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e')) 
	ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
	ax.set_xlim((0, surrgate_intervalres.shape[0])) 
	ax.set_xlabel('Surrogate interval', fontsize=14) 
	ax.set_ylabel('RMSE', fontsize=14)
	#ax.yaxis.set_label_coords(-0.1,1.02)
	#ax.yaxis.set_ticks_position('bottom')

	#plt.xlabel('Surrogate Interval ID', fontsize=18)
	#plt.ylabel(' RMSE', fontsize=18)
	#plt.legend(loc='upper right') 
	plt.savefig(path_db+'/surrogate_intervalerror.png' ) 


	surr_meantrain = np.mean(surrgate_intervalres)
	surr_stdtrain = np.std(surrgate_intervalres)



	outres = open(path+'/result.txt', "a+") 
	outres_db = open(path_db+'/result.txt', "a+") 

	resultingfile = open(problemfolder+'/master_result_file.txt','a+')  
	resultingfile_db = open( problemfolder_db+'/master_result_file.txt','a+') 

	xv = name+'_'+ str(run_nb) 

	print (  acc_tr, acctr_max, acc_tes, acctes_max)  
	allres =  np.asarray([ problem, NumSample, maxtemp, swap_interval, surrogate_intervalratio, surrogate_prob,  use_langevin_gradients, learn_rate, acc_tr, acctr_std, acctr_max, acc_tes, acctest_std, acctes_max, swap_perc, accept, rmse_surr,  timetotal]) 
	 
	np.savetxt(outres_db,  allres   , fmt='%1.2f', newline=' '  )   
	np.savetxt(resultingfile_db,   allres   , fmt='%1.2f',  newline=' ' ) 
	np.savetxt(resultingfile_db, (surr_meantrain, surr_stdtrain), fmt='%1.2e',   newline=' ' )   
	np.savetxt(resultingfile_db, [xv]   ,  fmt="%s", newline=' \n' )   

	np.savetxt(outres,  allres   , fmt='%1.2f', newline=' '  )   
	np.savetxt(resultingfile,   allres   , fmt='%1.2f',  newline=' ' ) 
	np.savetxt(resultingfile, (surr_meantrain, surr_stdtrain)  ,  fmt='%1.2e',  newline=' ' )  
	np.savetxt(resultingfile, [xv]   ,  fmt="%s", newline=' \n' )   



	plt.plot(acc_train,  label='Test' )
	plt.plot(acc_test,   label='Train' ) 
	plt.xlabel('Samples', fontsize=14)
	plt.ylabel(' Accuracy (%)', fontsize=14)
	plt.legend(loc='upper right')

	plt.savefig(path+'/acc_samples.png') 
	 
	plt.savefig(path_db+'/acc_samples.png') 
	plt.clf()	

	print(rmse_train.shape)
	plt.plot(rmse_train[:, 0],  label='Train')
	plt.plot(rmse_test[:, 0],   label='Test') 

	plt.xlabel('Samples', fontsize=14)
	plt.ylabel(' RMSE', fontsize=14)
	plt.legend(loc='upper right')

	plt.savefig(path+'/rmse_samples.png') 
	 
	plt.savefig(path_db+'/rmse_samples.png') 
	plt.clf()




	likelihood = likelihood_rep[:,0] # just plot proposed likelihood
	likelihood = np.asarray(np.split(likelihood, num_chains))

# Plots
	plt.plot(likelihood.T ) 
	plt.xlabel('Samples', fontsize=14)
	plt.ylabel('Log-Likelihood', fontsize=14)

	plt.legend(loc='upper left')
	plt.savefig(path+'/likelihood.png') 
	plt.savefig(path_db+'/likelihood.png')
	plt.clf()

	list_true = np.asarray(np.split(surr_list[:,0], num_chains))
	list_surrogate = np.asarray(np.split(surr_list[:,1], num_chains))
	list_true = list_true.T 
	list_surrogate = list_surrogate.T
 

	plt.plot(list_true[0:pt_samples, 0:3  ],   label='True' )
	plt.plot(list_surrogate[0:pt_samples, 0:3 ], '.',   label='Surrogate' )  
	plt.xlabel('Samples', fontsize=14)
	plt.ylabel('Log-Likelihood', fontsize=14)
	#plt.xticks(fontsize=14)

	plt.legend(loc='upper right')
	plt.savefig(path+'/surr_likelihood.png') 
	plt.savefig(path_db+'/surr_likelihood.png')
	plt.clf()

	plt.plot(list_true[pt_samples+1:, 0:3  ],   label='True' )
	plt.plot(list_surrogate[pt_samples+1:, 0:3 ], '.',   label='Surrogate' )  
	plt.xlabel('Samples', fontsize=14)
	plt.ylabel('Likelihood', fontsize=14)

	plt.legend(loc='upper right')
	plt.savefig(path+'/surr_likelihood_.png') 
	plt.savefig(path_db+'/surr_likelihood_.png')
	plt.clf()












	'''x_index = np.where(fx_train==np.inf)  
	fx_train = np.delete(fx_train, x_index, axis = 0) 

	print(fx_train, ' fx_train')
	 


	fx_train_mean = fx_train.mean(axis=1)


	print(fx_train_mean, ' fx_train mean')


	fx_train_5th = np.percentile(fx_train, 5, axis=1)
	fx_train_95th= np.percentile(fx_train, 95, axis=1)

	plt.plot(fx_train_mean,'.', label='Pred Train') 
	plt.plot(fx_train_5th,'.', label='5thTrain') 
	plt.plot(fx_train_95th,'.', label='95th Train')
	plt.plot(y_train, '.', label='Actual Train') 
	plt.legend(loc='upper right')

	plt.title("Pred. Train")
	plt.savefig(path+'/fxtrain_samples.png') 
	plt.clf()	'''




	#mpl_fig = plt.figure()
	#ax = mpl_fig.add_subplot(111)

	# ax.boxplot(pos_w)

	# ax.set_xlabel('[W1] [B1] [W2] [B2]')
	# ax.set_ylabel('Posterior')

	# plt.legend(loc='upper right')

	# plt.title("Boxplot of Posterior W (weights and biases)")
	# plt.savefig(path+'/w_pos.png')
	# plt.savefig(path+'/w_pos.svg', format='svg', dpi=600)

	# plt.clf()
	#dir()
	gc.collect()
	outres.close()

	outres_db.close()

	resultingfile.close()

	resultingfile_db.close()


	time.sleep(5)

if __name__ == "__main__": main()
