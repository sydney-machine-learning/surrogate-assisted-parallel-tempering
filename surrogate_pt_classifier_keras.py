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


from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.objectives import MSE, MAE
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.models import load_model

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

	def BackwardPass(self, Input, desired):
		out_delta = (desired - self.out).dot(self.out.dot(1 - self.out))
		hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))
		print(self.B2.shape)
		self.W2 += (self.hidout.T.reshape(self.Top[1],1).dot(out_delta) * self.lrate)
		self.B2 += (-1 * self.lrate * out_delta)
		self.W1 += (Input.T.reshape(self.Top[0],1).dot(hid_delta) * self.lrate)
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
		w2 = self.W2.ravel()
		w = np.concatenate([w1, w2, self.B1, self.B2])
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
		return fx, prob

class surrogate: #General Class for surrogate models for predicting likelihood given the weights
	
	#def __init__(self, model, X, Y, path): 
	#	X = np.delete(X, indices, axis=0)
	#	Y = np.delete(Y,indices, axis=0) 
	#	self.Y = Y
	#	self.X = X 
	#	self.path = path
	#	if model=="gp":
	#		self.model_id = 1
	#	else:
	#		if model == "nn":
	#			self.model_id = 2
	#		else:
	#			print("Invalid Model!")

	def __init__(self, model, X, Y, min_X, max_X, min_Y , max_Y, path):
		
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
		if self.model_id is 1:
			ker = GPy.kern.Matern52(input_dim = self.X.shape[1], lengthscale = 1., ARD=True) + GPy.kern.White(self.X.shape[1])
			gp_load = GPy.models.GPRegression(X_train,y_train,ker)
			gp_load.update_model(False)
			gp_load.initialize_parameter()
			fname = self.path + '/gp_params'
			try:
				gp_load[:] = np.load('%s_gp_params.npy'%(fname))
			except EnvironmentError:
				print('Training Model for the first time')
			gp_load.update_model(True)
			gp_load.optimize_restarts(messages = False, num_restarts = 2)
			y_pred = gp_load.predict(X_test)[0]

			mse = mean_squared_error(y_test.ravel(), y_pred.ravel())
			r2 = r2_score(y_test.ravel(), y_pred.ravel())
			
			np.save('%s_gp_params.npy' % (fname), gp_load.param_array)
			print("After Training: MSE = ",mse," R sqaured score = ",r2)
		if self.model_id is 2:
			#Neural Network for prediction
			try:
				[net]= pickle.load(open(self.path+'/nn_params.pckl','rb+'))
			except EnvironmentError:
				net = MLPRegressor(hidden_layer_sizes=(350,17,),activation='relu',solver='adam',alpha=0.025, max_iter = 1000)
			net.fit(X_train,y_train.ravel())
			y_pred = net.predict(X_test)
			print(y_pred)
			mse = mean_squared_error(y_test.ravel(), y_pred.ravel())/(max(y_test)-min(y_test))
			r2 = r2_score(y_test.ravel(), y_pred.ravel())
			f = open(self.path+'/nn_params.pckl','wb+')
			pickle.dump([net], f)
			f.close()
			print("After Training: MSE = ",mse," R squared score = ",r2)


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
						print (' Tried to load file : ', self.path+'/model_krnn_%s_.h5'%(self.model_signature-1))
						self.krnn = load_model(self.path+'/model_krnn_%s_.h5'%(model_signature-1))
						break
					except EnvironmentError as e:
						pass
						# print(e.errno)
						# time.sleep(1)
						# print ('ERROR in loading latest surrogate model, loading previous one in TRAIN')
				
			early_stopping = EarlyStopping(monitor='val_loss', patience=10)
			self.krnn.compile(loss='mse', optimizer='adam', metrics=['mse'])
			train_log = self.krnn.fit(X_train, y_train.ravel(), batch_size=10, epochs=100, validation_split=0.1, verbose=2, callbacks=[early_stopping])

			scores = self.krnn.evaluate(X_test, y_test.ravel(), verbose = 0)
			print("%s: %.5f" % (self.krnn.metrics_names[1], scores[1]))
		
			self.krnn.save(self.path+'/model_krnn_%s_.h5' %self.model_signature)
			print("Saved model to disk  ", self.model_signature)

			plt.plot(train_log.history["loss"], label="loss")
			plt.plot(train_log.history["val_loss"], label="val_loss")
			# plt.show()
			plt.savefig(self.path+'/%s_0.png'%(self.model_signature))

			results = np.array([scores[1]])
			make_directory(self.path+"/prediction_benchmark_data")
			np.savetxt(self.path+'/Train_metrics.csv', results)
			np.savetxt(self.path+'/prediction_benchmark_data/X_train.csv', X_train)
			np.savetxt(self.path+'/prediction_benchmark_data/Y_train.csv', y_train)
			np.savetxt(self.path+'/prediction_benchmark_data/X_test.csv', X_test)
			np.savetxt(self.path+'/prediction_benchmark_data/Y_test.csv', y_test)

	# def predict(self, X_load):
	# 	if self.model_id == 1:
	# 		fname = self.path + '/gp_params'
	# 		ker = GPy.kern.Matern52(input_dim = self.X.shape[1], lengthscale = 1., ARD=True) + GPy.kern.White(self.X.shape[1])
	# 		gp_load = GPy.models.GPRegression(self.X, self.Y, initialize=False,kernel=ker)
	# 		gp_load.update_model(False)
	# 		gp_load.initialize_parameter()
	# 		gp_load[:] = np.load('%s_gp_params.npy'%(fname))
	# 		gp_load.update_model(True)
	# 		return gp_load.predict(X_load)[0].ravel()[0]
	# 	if self.model_id == 2:
	# 		f = open(self.path+'/nn_params.pckl','rb+')
	# 		[net]= pickle.load(f)
	# 		f.close()
	# 		# print(net.predict(X_load))
	# 		# print(X_load)
	# 		return (net.predict(X_load)[0])


	def predict(self, X_load, initialized):
		if self.model_id == 1:
			ker = GPy.kern.Matern52(input_dim = self.X.shape[1], lengthscale = 1., ARD=True) + GPy.kern.White(self.X.shape[1])
			gp_load = GPy.models.GPRegression(self.X, self.Y, initialize=False,kernel=ker)
			gp_load.update_model(False)
			gp_load.initialize_parameter()
			gp_load[:] = np.load('%s_gp_params.npy'%(self.path + '/gp_params'))
			gp_load.update_model(True)
			
			return gp_load.predict(X_load)[0].ravel()[0]
		
		if self.model_id == 2:
			f = open(self.path+'/nn_params.pckl','rb+')
			[net, self.max_Y, self.min_Y]= pickle.load(f)
			f.close()
			print('Output', net.predict(X_load)[0])
			nn_prediction = net.predict(X_load)[0]
			prediction =  net.predict(X_load)[0]*(self.max_Y[0,0]-self.min_Y[0,0]) + self.min_Y[0,0]
			
			return prediction

		if self.model_id == 3:
			
			if initialized == False:
				model_sign = np.loadtxt(self.path+'/model_signature.txt')
				self.model_signature = model_sign 
				while True:
					try:
						self.krnn = load_model(self.path+'/model_krnn_%s_.h5'%self.model_signature)
						print (' Tried to load file : ', self.path+'/model_krnn_%s_.h5'%self.model_signature)
						break
					except EnvironmentError as e:
						pass
						# IN CASE OF JUGAAR USE THIS:
						# time.sleep(2)
						# krnn = load_model(self.path+'/model_krnn_%s_.h5'%(self.model_signature-1))
			
				self.krnn.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
				# krnn_prediction = krnn.predict(X_load)[0]
				prediction = -1.0
			
			else:
				krnn_prediction = self.krnn.predict(X_load)[0]
				prediction = krnn_prediction*(self.max_Y[0,0]-self.min_Y[0,0]) + self.min_Y[0,0]

			return prediction


class ptReplica(multiprocessing.Process):

	def __init__(self, w,  minlim_param, maxlim_param, samples, traindata, testdata, topology, burn_in, temperature, swap_interval, path, parameter_queue, main_process,event,surrogate_parameterqueue,surrogate_interval,surrogate_prob,surrogate_start,surrogate_resume):
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


	def rmse(self, pred, actual):
		return np.sqrt(((pred-actual)**2).mean())

	def accuracy(self,pred,actual):
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
		return [lhood/self.temperature, fx, rmse]

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
		surrogate_list = np.zeros((samples,1))
		fxtrain_samples = np.ones((samples, trainsize)) #Output of regression FNN for training samples
		fxtest_samples = np.ones((samples, testsize)) #Output of regression FNN for testing samples
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
		[likelihood, pred_train, rmsetrain] = self.likelihood_func(fnn, self.traindata, w)
		[_, pred_test, rmsetest] = self.likelihood_func(fnn, self.testdata, w)
		#Beginning Sampling using MCMC RANDOMWALK
		accept_list = open(self.path+'/acceptlist_'+str(int(self.temperature*10))+'.txt', "a+")
		trainacc = 0
		testacc=0

		prop_list = np.zeros((samples,w_proposal.size))
		likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood
		likeh_list[0,:] = [-100, -100] # to avoid prob in calc of 5th and 95th percentile later
		surg_likeh_list = np.zeros((samples,2))







		#Surrogate Init








		surrogate_counter = 0
		
		for i in range(samples-1):

			timer1 = time.time()
			is_true_lhood = True
 
			 
			#is_surr_lhood = False 

			# Update by perturbing all the  parameters via "random-walk" sampler 

			w_proposal = np.random.normal(w, step_w, w_size)  


			#[likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(fnn, self.traindata, w_proposal)
 

			ku = random.uniform(0,1)
			burnsamples = int(self.samples * self.burn_in)
			if ku<self.surrogate_prob and i>=self.surrogate_interval+1 and i>burnsamples:
				if surrogate_model == None:
					minmax = np.loadtxt(self.path+'/surrogate/minmax.txt')
					self.minY[0,0] = minmax[0]
					self.maxY[0,0] = minmax[1]
					surrogate_model = Surrogate("krnn",surrogate_X.copy(),surrogate_Y.copy(), self.minlim_param, self.maxlim_param, self.minY, self.maxY, self.path)
					surrogate_likelihood = surrogate_model.predict(w_proposal.reshape(1,w_proposal.shape[0]),False)
					surrogate_likelihood = surrogate_likelihood *(1.0/self.temperature)
						
				elif self.surrogate_init == 0.0:
					surrogate_likelihood = surrogate_model.predict(w_proposal.reshape(1,w_proposal.shape[0]), False)
					surrogate_likelihood = surrogate_likelihood *(1.0/self.temperature)
				else:
					surrogate_likelihood = surrogate_model.predict(w_proposal.reshape(1,w_proposal.shape[0]), True)
					surrogate_likelihood = surrogate_likelihood *(1.0/self.temperature)

				surg_likeh_list[i+1,0] = likelihood_proposal
				surg_likeh_list[i+1,1] = surrogate_likelihood
				print ('\nSample : ', i, ' Chain :', self.temperature, ' -A', likelihood_proposal, ' vs. P ', surrogate_likelihood)
				surrogate_counter += 1

				likelihood_proposal = surrogate_likelihood

			else: 

				[likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(fnn, self.traindata, w_proposal)
				surg_likeh_list[i+1,0] = likelihood_proposal
				surg_likeh_list[i+1,1] = np.nan

			prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal)  # takes care of the gradients
			
			diff_likelihood = likelihood_proposal - likelihood

			diff_prior = prior_prop - prior_current
			try:
				mh_prob = min(1, math.exp(diff_likelihood))
			except OverflowError as e:
				mh_prob = 1

			u = random.uniform(0, 1)
			
			prop_list[i+1,] = w_proposal	
			likeh_list[i+1,0] = likelihood_proposal

			if u < mh_prob:
				naccept  =  naccept + 1
				likelihood = likelihood_proposal
				prior_current = prior_prop
				w = w_proposal
				print (i, self.temperature, likelihood, rmsetrain, rmsetest,  'accepted')
				accept_list.write('{} {} {} {} {} {} {}\n'.format(self.temperature,naccept, i, rmsetrain, rmsetest, diff_likelihood, diff_likelihood + diff_prior))
				pos_w[i + 1,] = w_proposal
				s_pos_w[i+1,] = w_proposal
				if is_true_lhood == True:
					lhood_list[i+1,] = (likelihood*self.temperature)
				else:
					lhood_list[i+1,] = np.inf
				fxtrain_samples[i + 1,] = pred_train
				fxtest_samples[i + 1,] = pred_test
				rmse_train[i + 1,] = rmsetrain
				rmse_test[i + 1,] = rmsetest
				acc_train[i+1,] = self.accuracy(pred_train, y_train)
				acc_test[i+1,] = self.accuracy(pred_test, y_test)

			else:
				accept_list.write('{} x {} {} {} {} {}\n'.format(self.temperature, i, rmsetrain, rmsetest, likelihood, diff_likelihood + diff_prior))
				pos_w[i+1,] = pos_w[i,]
				s_pos_w[i + 1,] = w_proposal
				if is_true_lhood == True:
					lhood_list[i+1,] = (likelihood_proposal*self.temperature)
				else:
					lhood_list[i+1,] = np.inf 
				fxtrain_samples[i + 1,] = fxtrain_samples[i,]
				fxtest_samples[i + 1,] = fxtest_samples[i,]
				rmse_train[i + 1,] = rmse_train[i,]
				rmse_test[i + 1,] = rmse_test[i,]
				acc_train[i+1,] = acc_train[i,]
				acc_test[i+1,] = acc_test[i,]
			#SWAPPING PREP
			if i%self.swap_interval == 0:
				param = np.concatenate([w, np.asarray([eta]).reshape(1), np.asarray([likelihood]),np.asarray([self.temperature]),np.asarray([i])])
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
			#SURROGATE TRAINING
			# if (i%self.surrogate_interval == 0) and (i!=0):
			# 	#Train the surrogate with the posteriors and likelihood
			# 	surrogate_X, surrogate_Y = s_pos_w[i+1-self.surrogate_interval:i,:],lhood_list[i+1-self.surrogate_interval:i,:]
			# 	param = np.concatenate([s_pos_w[i+1-self.surrogate_interval:i,:],lhood_list[i+1-self.surrogate_interval:i,:]],axis=1)
			# 	self.surrogate_parameterqueue.put(param)
			# 	self.surrogate_start.set()
			# 	self.surrogate_resume.wait()

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
					surrogate_model = surrogate("krnn", dummy_X, dummy_Y, self.minlim_param, self.maxlim_param, self.minY, self.maxY, self.path)
				
				self.surrogate_init = surrogate_model.predict(w_proposal.reshape(1,w_proposal.shape[0]), False)				
				print("Surrogate init ", self.surrogate_init , " - should be -1")



		param = np.concatenate([w, np.asarray([eta]).reshape(1), np.asarray([likelihood]),np.asarray([self.temperature]),np.asarray([i])])
		#print('SWAPPED PARAM',self.temperature,param)
		self.parameter_queue.put(param)
		param = np.concatenate([s_pos_w[i-self.surrogate_interval:i,:],lhood_list[i-self.surrogate_interval:i,:]],axis=1)
		self.surrogate_parameterqueue.put(param)
		make_directory(self.path+'/results')
		make_directory(self.path+'/posterior')

		print ((naccept*100 / (samples * 1.0)), '% was accepted')
		accept_ratio = naccept / (samples * 1.0) * 100
		fig = plt.figure()
		plt.plot(acc_train, label="Train")
		plt.plot(acc_test, label="Test")
		plt.legend()
		plt.savefig(self.path+'/accuracy'+str(self.temperature)+'.png')
		plt.close()
		########PLOTTING SURROGATES###############################################
		# fig = plt.figure()
		# plt.plot(lhood_list[self.surrogate_interval+2:samples-1], label="True")
		# plt.plot(surrogate_list[self.surrogate_interval+2:samples-1], label="Predict")
		# plt.legend()
		# plt.savefig(self.path+'/surrogate'+str(self.temperature)+'.pdf')
		# plt.close()
		#SAVING PARAMETERS
		file_name = self.path+'/posterior/pos_w_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,pos_w ) 
		file_name = self.path+'/posterior/fxtrain_samples_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, fxtrain_samples, fmt='%1.2f')
		file_name = self.path+'/posterior/fxtest_samples_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, fxtest_samples, fmt='%1.2f')		
		file_name = self.path+'/posterior/rmse_test_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, rmse_test, fmt='%1.2f')		
		file_name = self.path+'/posterior/rmse_train_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, rmse_train, fmt='%1.2f')
		file_name = self.path+'/posterior/acc_train_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, acc_train, fmt='%1.2f')	
		file_name = self.path+'/posterior/acc_test_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, acc_test, fmt='%1.2f')			
		file_name = self.path + '/posterior/accept_list_chain_' + str(self.temperature) + '_accept.txt'
		np.savetxt(file_name, [accept_ratio], fmt='%1.2f')

		self.signal_main.set()
		self.surrogate_start.set()

class ParallelTempering:

	def __init__(self, traindata, testdata, topology, num_chains, maxtemp, NumSample, swap_interval, surrogate_interval, surrogate_prob, path):
		#FNN Chain variables
		self.traindata = traindata
		self.testdata = testdata
		self.topology = topology
		self.num_param = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
		#Parallel Tempering variables
		self.swap_interval = swap_interval
		self.path = path
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

	def default_beta_ladder(self, ndim, ntemps, Tmax): #https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
		"""
		Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
		arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
		Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
		this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
		<http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
		``ntemps`` is also specified.
		:param ndim:
			The number of dimensions in the parameter space.
		:param ntemps: (optional)
			If set, the number of temperatures to generate.
		:param Tmax: (optional)
			If set, the maximum temperature for the ladder.
		Temperatures are chosen according to the following algorithm:
		* If neither ``ntemps`` nor ``Tmax`` is specified, raise an exception (insufficient
		  information).
		* If ``ntemps`` is specified but not ``Tmax``, return a ladder spaced so that a Gaussian
		  posterior would have a 25% temperature swap acceptance ratio.
		* If ``Tmax`` is specified but not ``ntemps``:
		  * If ``Tmax = inf``, raise an exception (insufficient information).
		  * Else, space chains geometrically as above (for 25% acceptance) until ``Tmax`` is reached.
		* If ``Tmax`` and ``ntemps`` are specified:
		  * If ``Tmax = inf``, place one chain at ``inf`` and ``ntemps-1`` in a 25% geometric spacing.
		  * Else, use the unique geometric spacing defined by ``ntemps`` and ``Tmax``.
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

		print(self.minlim_param)


		w = np.random.randn(self.num_param)
		
		for i in range(0, self.num_chains):
			self.chains.append(ptReplica(w,  self.minlim_param, self.maxlim_param, self.NumSamples,self.traindata,self.testdata,self.topology,self.burn_in,self.temperatures[i],self.swap_interval,self.path,self.parameter_queue[i],self.wait_chain[i],self.event[i],self.surrogate_parameterqueues[i],self.surrogate_interval,self.surrogate_prob,self.surrogate_start_events[i],self.surrogate_resume_events[i]))

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
				print ("Swapped")
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
			self.minY[0,i] = min_Y
			self.maxY[0,i] = max_Y 

		self.model_signature += 1.0
		make_directory(self.path+'/surrogate')
		if self.model_signature == 1.0:
			np.savetxt(self.path+'/surrogate/minmax.txt',[self.minY[0, 0], self.maxY[0, 0]])
		
		np.savetxt(self.path+'/surrogate/model_signature.txt', [self.model_signature])
		
		Y= self.normalize_likelihood(Y)
		indices = np.where(Y==np.inf)[0]
		X = np.delete(X, indices, axis=0)
		Y = np.delete(Y,indices, axis=0)
		surrogate_model = surrogate("krnn", X , Y , self.minlim_param, self.maxlim_param, self.minY, self.maxY, self.path)
		surrogate_model.train(self.model_signature)


	def normalize_likelihood(self, Y):
		for i in range(Y.shape[1]):
			if self.model_signature == 1.0:
				min_Y = min(Y[:,i])
				max_Y = max(Y[:,i])
				# self.minY[0,i] = 1 #For Tau Squared
				# self.maxY[0,i] = max_Y 
				
				self.maxY[0,i] = max_Y
				self.minY[0,i] = min_Y

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
		x_test = np.linspace(0,1,num=self.testdata.shape[0])
		x_train = np.linspace(0,1,num=self.traindata.shape[0])
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


			#print('surrogate')
			
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
					if params is not None:
						self.surrchain_queue.task_done()
						#print(k,'No Process')
						break
				if params is not None:
					all_param = np.asarray(params if not ('all_param' in locals()) else np.concatenate([all_param,params],axis=0))
			if ('all_param' in locals()):
				if all_param.shape == (self.num_chains*(self.surrogate_interval-1),self.num_param+1):
					self.surrogate_trainer(all_param)
					del all_param
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
		#GETTING DATA
		burnin = int(self.NumSamples*self.burn_in)
		pos_w = np.zeros((self.num_chains,self.NumSamples - burnin, self.num_param))
		fxtrain_samples = np.zeros((self.num_chains,self.NumSamples - burnin, self.traindata.shape[0]))
		rmse_train = np.zeros((self.num_chains,self.NumSamples - burnin))
		acc_train = np.zeros((self.num_chains,self.NumSamples - burnin))
		fxtest_samples = np.zeros((self.num_chains,self.NumSamples - burnin, self.testdata.shape[0]))
		rmse_test = np.zeros((self.num_chains,self.NumSamples - burnin))
		acc_test = np.zeros((self.num_chains,self.NumSamples - burnin))
		accept_ratio = np.zeros((self.num_chains,1))

		for i in range(self.num_chains): #Loading Files for output
			file_name = self.path+'/posterior/pos_w_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			pos_w[i,:,:] = dat[burnin:,:] 
			file_name = self.path+'/posterior/fxtrain_samples_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			fxtrain_samples[i,:,:] = dat[burnin:,:]
			file_name = self.path+'/posterior/fxtest_samples_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			fxtest_samples[i,:,:] = dat[burnin:,:]	
			file_name = self.path+'/posterior/rmse_test_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			rmse_test[i,:] = dat[burnin:]		
			file_name = self.path+'/posterior/rmse_train_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			rmse_train[i,:] = dat[burnin:]
			file_name = self.path+'/posterior/acc_train_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			acc_train[i,:] = dat[burnin:]
			file_name = self.path+'/posterior/acc_test_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			acc_test[i,:] = dat[burnin:]
			file_name = self.path + '/posterior/accept_list_chain_' + str(self.temperatures[i]) + '_accept.txt'
			dat = np.loadtxt(file_name)
			accept_ratio[i,:] = dat
		pos_w = pos_w.transpose(2,0,1).reshape(self.num_param,-1)
		accept_total = np.sum(accept_ratio)/self.num_chains
		fx_train = fxtrain_samples.reshape(self.num_chains*(self.NumSamples - burnin), self.traindata.shape[0])
		rmse_train = rmse_train.reshape(self.num_chains*(self.NumSamples - burnin), 1)
		acc_train = acc_train.reshape(self.num_chains*(self.NumSamples - burnin), 1)
		fx_test = fxtest_samples.reshape(self.num_chains*(self.NumSamples - burnin), self.testdata.shape[0])
		rmse_test = rmse_test.reshape(self.num_chains*(self.NumSamples - burnin), 1)
		acc_test = acc_test.reshape(self.num_chains*(self.NumSamples - burnin), 1)
		# for s in range(self.num_param):  
		# 	self.plot_figure(pos_w[s,:], 'pos_distri_'+str(s)) 
		print("accuracies", max(acc_train), max(acc_test))
		print("NUMBER OF SWAPS =", self.num_swap)
		print("SWAP ACCEPTANCE = ", [0 if self.total_swap_proposals is 0 else self.num_swap*100/self.total_swap_proposals]," %")
		return (pos_w, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_total)

def make_directory (directory): 
	if not os.path.exists(directory):
		os.makedirs(directory)

def main():

	for i in range(7, 8):
		problem = i
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
		if problem == 2: #IRIS
			data  = np.genfromtxt('DATA/iris.csv',delimiter=';')
			classes = data[:,4].reshape(data.shape[0],1)-1
			features = data[:,0:4]
			separate_flag = True
			name = "iris"
			hidden = 12
			ip = 4 #input
			output = 3
		if problem == 3: #Ionosphere
			traindata = np.genfromtxt('DATA/Ions/Ions/ftrain.csv',delimiter=',')[:,:-1]
			testdata = np.genfromtxt('DATA/Ions/Ions/ftest.csv',delimiter=',')[:,:-1]
			name = "Ionosphere"
			hidden = 50
			ip = 34 #input
			output = 2
		if problem == 4: #Cancer
			traindata = np.genfromtxt('DATA/Cancer/ftrain.txt',delimiter=' ')[:,:-1]
			testdata = np.genfromtxt('DATA/Cancer/ftest.txt',delimiter=' ')[:,:-1]
			name = "Cancer"
			hidden = 12
			ip = 9 #input
			output = 2
		if problem == 5: #Wine Quality White
			data  = np.genfromtxt('DATA/winequality-white.csv',delimiter=';')
			data = data[1:,:] #remove Labels
			classes = data[:,11].reshape(data.shape[0],1)
			features = data[:,0:11]
			separate_flag = True
			name = "winequality-white"
			hidden = 50
			ip = 11 #input
			output = 10
		if problem == 6: #Bank additional
			data = np.genfromtxt('DATA/Bank/bank-processed.csv',delimiter=';')
			classes = data[:,20].reshape(data.shape[0],1)
			features = data[:,0:20]
			separate_flag = True
			name = "bank-additional"
			hidden = 50
			ip = 20 #input
			output = 2
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
		if problem == 8: #Chess
			data  = np.genfromtxt('DATA/chess.csv',delimiter=';')
			classes = data[:,6].reshape(data.shape[0],1)
			features = data[:,0:6]
			separate_flag = True
			name = "chess"
			hidden = 25
			ip = 6 #input
			output = 18
		###############################
		#THESE ARE THE HYPERPARAMETERS#
		###############################
		topology = [ip, hidden, output]

		NumSample = 2000 
		maxtemp = 20 
		swap_ratio = 0.01
		num_chains = 10
		swap_interval = int(swap_ratio * (NumSample/num_chains)) #how ofen you swap neighbours
		burn_in = 0.2
		surrogate_interval = 20  
		surrogate_prob = 0.5




		problemfolder = 'SydneyResults/'

			#path = "SydneyResults/"+name+"_results_"+str(NumSample) 
		#make_directory(path)
		#print(path)


		fname = ""
		run_nb = 0
		while os.path.exists(problemfolder +name+'_%s' % (run_nb)):
			run_nb += 1
		if not os.path.exists(problemfolder +name+'_%s' % (run_nb)):
			os.makedirs(problemfolder +name+'_%s' % (run_nb))
			path = (problemfolder +name+'_%s' % (run_nb))

		#fname = ('sampleresults')
  
		#make_directory((fname + '/posterior/pos_parameters')) 


		#make_directory('SydneyResults')
		resultingfile = open( path+'/master_result_file.txt','a+')
		expfile = open( path+'/expdesign_file.txt','w')














		###############################
		if surrogate_interval < swap_interval:
			surrogate_interval = swap_interval
		if surrogate_interval%swap_interval!=0:
			surrogate_interval = surrogate_interval + swap_interval - surrogate_interval%swap_interval
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

		
		timer = time.time()
		#path = "SydneyResults/"+name+"_results_"+str(NumSample)+"_"+str(maxtemp)+"_"+str(num_chains)+"_"+str(swap_ratio)+"_"+str(surrogate_interval)+"_"+str(surrogate_prob)
		
	
		pt = ParallelTempering(traindata, testdata, topology, num_chains, maxtemp, NumSample, swap_interval, surrogate_interval, surrogate_prob, path)



 


		pt.initialize_chains(  burn_in)

		pos_w, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_total = pt.run_chains()

		print ('Successfully Regressed')
		print (accept_total, '% total accepted')

		timer2 = time.time()
		print ((timer2 - timer), 'sec time taken')

		#PLOTS
		fx_mu = fx_test.mean(axis=0)
		fx_high = np.percentile(fx_test, 95, axis=0)
		fx_low = np.percentile(fx_test, 5, axis=0)

		fx_mu_tr = fx_train.mean(axis=0)
		fx_high_tr = np.percentile(fx_train, 95, axis=0)
		fx_low_tr = np.percentile(fx_train, 5, axis=0)

		rmse_tr = np.mean(rmse_train[:])
		rmsetr_std = np.std(rmse_train[:])
		rmse_tes = np.mean(rmse_test[:])
		rmsetest_std = np.std(rmse_test[:])

		outres = open(path+'/result.txt', "a+")
		np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_total), fmt='%1.5f')
		print (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std)

		np.savetxt(resultingfile,(NumSample, maxtemp, swap_ratio, num_chains, rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_total))
		ytestdata = testdata[:, ip]
		ytraindata = traindata[:, ip]

		plt.plot(x_test, ytestdata,'.', label='actual')
		plt.plot(x_test, fx_mu, '.', label='pred. (mean)')
		plt.plot(x_test, fx_low, '.', label='pred.(5th percen.)')
		plt.plot(x_test, fx_high, '.', label='pred.(95th percen.)')
		plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Plot of Test Data vs MCMC Uncertainty ")
		plt.savefig(path+'/restest.png')
		plt.savefig(path+'/restest.svg', format='svg', dpi=600)
		plt.clf()
		# -----------------------------------------
		plt.plot(x_train, ytraindata, '.',label='actual')
		plt.plot(x_train, fx_mu_tr, '.',label='pred. (mean)')
		plt.plot(x_train, fx_low_tr, '.',label='pred.(5th percen.)')
		plt.plot(x_train, fx_high_tr, '.',label='pred.(95th percen.)')
		plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Plot of Train Data vs MCMC Uncertainty ")
		plt.savefig(path+'/restrain.png')
		plt.savefig(path+'/restrain.svg', format='svg', dpi=600)
		plt.clf()

		mpl_fig = plt.figure()
		ax = mpl_fig.add_subplot(111)

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
		os.remove(path+'/nn_params.pckl')
	resultingfile.close()

if __name__ == "__main__": main()
