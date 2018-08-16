#Main Contributer: Danial Azam  Email: dazam92@gmail.com

import multiprocessing
import os
import sys
import time
import gc
import numpy as np
import random
import itertools
import operator
import math
# import cmocean as cmo
import copy
import collections
import fnmatch
import shutil
import plotly
import plotly.plotly as py

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import scipy
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.spatial import cKDTree

import pickle
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import normalize, MinMaxScaler

# mpl.use('Agg')
from pylab import rcParams
from copy import deepcopy
from scipy import special
from PIL import Image
from io import StringIO
from cycler import cycler
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from plotly.graph_objs import *
from plotly.offline.offline import _plot_html

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.objectives import MSE, MAE
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras.models import load_model

import GPy
np.random.seed(1)
plotly.offline.init_notebook_mode()
GPy.plotting.change_plotting_library('plotly')
plotly.tools.set_credentials_file(username='dazam92', api_key='cZseSTjG7EBkTasOs8Op')

class Surrogate: #General Class for surrogate models for predicting likelihood given the weights
	
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

	def train(self, model_signature):
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.10, random_state=42)
		self.model_signature = model_signature
		
		if self.model_id is 2:
			#Neural Network for prediction
			try:
				[net, _, _]= pickle.load(open(self.path+'/nn_params.pckl','rb+'))
			except EnvironmentError as e:
				print(os.strerror(e.errno))
				net = MLPRegressor(hidden_layer_sizes=(30,),activation='logistic',
					solver='adam', max_iter = 1000, alpha=0.025)
			net.fit(X_train,y_train.ravel())

			y_pred = net.predict(X_test)
			mse = mean_squared_error(y_test.ravel(), y_pred.ravel())
			r2 = r2_score(y_test.ravel(), y_pred.ravel())
			f = open(self.path+'/nn_params.pckl','wb+')
			pickle.dump([net, self.max_Y, self.min_Y], f)
			f.close()
			print("After Training: MSE = ",mse," R squared score = ",r2)

			results = np.array([mse, r2])
			np.savetxt(self.path+'/Train_metrics.csv', results)
			np.savetxt(self.path+'/prediction_benchmark_data/X_train.csv', X_train)
			np.savetxt(self.path+'/prediction_benchmark_data/Y_train.csv', y_train)
			np.savetxt(self.path+'/prediction_benchmark_data/X_test.csv', X_test)
			np.savetxt(self.path+'/prediction_benchmark_data/Y_test.csv', y_test)
			np.savetxt(self.path+'/prediction_benchmark_data/Y_pred.csv', y_pred)

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
						print ' Tried to load file : ', self.path+'/model_krnn_%s_.h5'%(self.model_signature-1)
						self.krnn = load_model(self.path+'/model_krnn_%s_.h5'%(model_signature-1))
						break
					except EnvironmentError as e:
						print(e.errno)
						time.sleep(1)
						print 'ERROR in loading latest surrogate model, loading previous one in TRAIN'
				
			early_stopping = EarlyStopping(monitor='val_loss', patience=10)
			self.krnn.compile(loss='mse', optimizer='adam', metrics=['mse'])
			train_log = self.krnn.fit(X_train, y_train.ravel(), batch_size=10, epochs=100, validation_split=0.1, verbose=2, callbacks=[early_stopping])

			scores = self.krnn.evaluate(X_test, y_test.ravel(), verbose = 0)
			print("%s: %.5f" % (self.krnn.metrics_names[1], scores[1]))
		
			self.krnn.save(self.path+'/model_krnn_%s_.h5' %self.model_signature)
			print "Saved model to disk  ", self.model_signature

			plt.plot(train_log.history["loss"], label="loss")
			plt.plot(train_log.history["val_loss"], label="val_loss")
			# plt.show()
			plt.savefig(self.path+'/%s_0.png'%(self.model_signature))

			results = np.array([scores[1]])
			np.savetxt(self.path+'/Train_metrics.csv', results)
			np.savetxt(self.path+'/prediction_benchmark_data/X_train.csv', X_train)
			np.savetxt(self.path+'/prediction_benchmark_data/Y_train.csv', y_train)
			np.savetxt(self.path+'/prediction_benchmark_data/X_test.csv', X_test)
			np.savetxt(self.path+'/prediction_benchmark_data/Y_test.csv', y_test)

		return 

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
						print ' Tried to load file : ', self.path+'/model_krnn_%s_.h5'%self.model_signature
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

class PtReplica(multiprocessing.Process):
	def __init__(self, num_param, vec_param, minlim_param, maxlim_param, stepratio_vec,
		likelihood_sed, swap_interval, sim_interval, simtime, samples, real_elev, real_erodep_pts,
		erodep_coords, filename, xmlinput, run_nb, temperature, burn_in, parameter_queue, event,
		main_proc, surrogate_parameterqueue, surrogate_interval,surrogate_prob,	surrogate_start,
		surrogate_resume):
		#MULTIPROCESSING VARIABLES
		multiprocessing.Process.__init__(self)
		self.processID = temperature
		self.parameter_queue = parameter_queue
		self.signal_main = main_proc
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
		self.burn_in = burn_in
		#BAYESLAND VARIABLES
		self.filename = filename
		self.input = xmlinput  
		self.simtime = simtime
		self.samples = samples
		self.run_nb = run_nb 
		self.num_param =  num_param
		self.font = 9
		self.width = 1 
		self.vec_param = np.asarray(vec_param)
		
		self.minlim_param = np.asarray(minlim_param)
		self.maxlim_param = np.asarray(maxlim_param)
		self.minY = np.zeros((1,1))
		self.maxY = np.ones((1,1))

		self.stepratio_vec = np.asarray(stepratio_vec)
		self.likelihood_sed =  likelihood_sed
		self.real_erodep_pts = real_erodep_pts
		self.erodep_coords = erodep_coords
		self.real_elev = real_elev 
		self.burn_in = burn_in
		self.sim_interval = sim_interval
		# if you want to have histograms of the chains during runtime in pos_variables folder NB: this has issues in Artimis
		self.runninghisto = False  
		#this is to ensure that the sediment likelihood is given more emphasis as it considers fewer points 
		self.sedscalingfactor = 50
		self.model_signature = 0
		self.surrogate_init = 0
		
	def interpolate_array(self, coords=None, z=None, dz=None):
		"""
		Interpolate the irregular spaced dataset from badlands on a regular grid.
		"""
		x, y = np.hsplit(coords, 2)
		dx = (x[1]-x[0])[0]

		nx = int((x.max() - x.min())/dx+1)
		ny = int((y.max() - y.min())/dx+1)
		xi = np.linspace(x.min(), x.max(), nx)
		yi = np.linspace(y.min(), y.max(), ny)

		xi, yi = np.meshgrid(xi, yi)
		xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
		XY = np.column_stack((x,y))

		tree = cKDTree(XY)
		distances, indices = tree.query(xyi, k=3)
		if len(z[indices].shape) == 3:
			z_vals = z[indices][:,:,0]
			dz_vals = dz[indices][:,:,0]
		else:
			z_vals = z[indices]
			dz_vals = dz[indices]

		zi = np.average(z_vals,weights=(1./distances), axis=1)
		dzi = np.average(dz_vals,weights=(1./distances), axis=1)
		onIDs = np.where(distances[:,0] == 0)[0]
		if len(onIDs) > 0:
			zi[onIDs] = z[indices[onIDs,0]]
			dzi[onIDs] = dz[indices[onIDs,0]]
		zreg = np.reshape(zi,(ny,nx))
		dzreg = np.reshape(dzi,(ny,nx))
		return zreg,dzreg

	def run_badlands(self, input_vector):

		model = badlandsModel()

		# Load the XmL input file
		model.load_xml(str(self.run_nb), self.input, muted=True)

		# Adjust erodibility based on given parameter
		model.input.SPLero = input_vector[1] 
		model.flow.erodibility.fill(input_vector[1])

		# Adjust precipitation values based on given parameter
		model.force.rainVal[:] = input_vector[0] 

		# Adjust m and n values
		if self.num_param > 2:
			model.input.SPLm = input_vector[2] 
			model.input.SPLn = input_vector[3] 

		if self.num_param > 4 and self.num_param == 6:  # will work for more parameters
			model.input.CDm = input_vector[4] # submarine diffusion
			model.input.CDa = input_vector[5] # aerial diffusion

		if self.num_param == 5: # Mountain
			#Round the input vector 
			#k=round(input_vector[4]*2)/2 #to closest 0.5 
			k=round(input_vector[4],1) #to closest 0.1

			#Load the current tectonic uplift parameters
			#tectonicValues=pandas.read_csv(str(model.input.tectFile[0]),sep=r'\s+',header=None,dtype=np.float).values

			#Adjust the parameters by our value k, and save them out
			#Writing files while multiprocessing is bad, so premake all options and just read the files
			newFile = "Examples/mountain/tect/uplift"+str(k)+".csv"
			#newtect = pandas.DataFrame(tectonicValues*k)
			#newtect.to_csv(newFile,index=False,header=False)

			#Update the model uplift tectonic values
			model.input.tectFile[0]=newFile
			#print(model.input.tectFile)

		elev_vec = collections.OrderedDict()
		erodep_vec = collections.OrderedDict()
		erodep_pts_vec = collections.OrderedDict()

		for x in range(len(self.sim_interval)):
			self.simtime = self.sim_interval[x]

			model.run_to_time(self.simtime, muted=True)
		 
			elev, erodep = self.interpolate_array(model.FVmesh.node_coords[:, :2], model.elevation,
			model.cumdiff)

			erodep_pts = np.zeros((self.erodep_coords.shape[0]))

			for count, val in enumerate(self.erodep_coords):
				erodep_pts[count] = erodep[val[0], val[1]]

			elev_vec[self.simtime] = elev
			erodep_vec[self.simtime] = erodep
			erodep_pts_vec[self.simtime] = erodep_pts

		return elev_vec, erodep_vec, erodep_pts_vec

	def likelihood_function(self,input_vector):

		pred_elev_vec, pred_erodep_vec, pred_erodep_pts_vec = self.run_badlands(input_vector)

		tausq = np.sum(np.square(pred_elev_vec[self.simtime] - self.real_elev))/self.real_elev.size
		tau_erodep =  np.zeros(self.sim_interval.size) 
		
		for i in range(self.sim_interval.size):
			
			tau_erodep[i] = (np.sum(np.square(pred_erodep_pts_vec[self.sim_interval[i]] - 
						self.real_erodep_pts[i]))/ self.real_erodep_pts.shape[1])
	
		likelihood_elev = (-0.5*np.log(2*math.pi*tausq)-0.5*np.square(pred_elev_vec[self.simtime]
					- self.real_elev) / tausq )
		likelihood_erodep = 0 
		
		if self.likelihood_sed  == True: 
			
			for i in range(1, self.sim_interval.size):
				
				likelihood_erodep += (np.sum(-0.5 * np.log(2 * math.pi * tau_erodep[i]) \
								- 0.5 * np.square(pred_erodep_pts_vec[self.sim_interval[i]] \
								- self.real_erodep_pts[i]) / tau_erodep[i])) # only considers point or core of erodep
				
			likelihood = np.sum(likelihood_elev) +  (likelihood_erodep * self.sedscalingfactor)

		else:
			likelihood = np.sum(likelihood_elev)

		return [likelihood *(1.0/self.temperature), pred_elev_vec, pred_erodep_pts_vec, likelihood]

	def rmse(self, pred, actual):
		# print('In RMSE ')
		return np.sqrt(((pred-actual)**2).mean())

	def accuracy(self,pred,actual):
		count = 0
		for i in range(pred.shape[0]):
			if pred[i] == actual[i]:
				count+=1
		return 100*(count/pred.shape[0])

	def prior_likelihood(self):
		# print 'We arent usint it'
		return 1

	def run(self):
		samples = self.samples
		count_list = [] 
		stepsize_vec = np.zeros(self.maxlim_param.size)
		span = (self.maxlim_param-self.minlim_param) 

		for i in range(stepsize_vec.size): # calculate the step size of each of the parameters
			stepsize_vec[i] = self.stepratio_vec[i] * span[i]

		v_proposal = self.vec_param # initial parameter values to be passed to Blackbox model 
		v_current = v_proposal # to give initial value of the chain
		initial_predicted_elev, initial_predicted_erodep, init_pred_erodep_pts_vec = self.run_badlands(v_current)
		[likelihood, predicted_elev,  pred_erodep_pts, surrogate_var] = self.likelihood_function(v_current )

		likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood
		likeh_list[0,:] = [-100, -100] # to avoid prob in calc of 5th and 95th percentile later
		surg_likeh_list = np.zeros((samples,2))

		count_list.append(0) # just to count number of accepted for each chain (replica)
		accept_list = np.zeros(samples)
		prev_accepted_elev = deepcopy(predicted_elev)
		prev_acpt_erodep_pts = deepcopy(pred_erodep_pts) 
		sum_elev = deepcopy(predicted_elev)
		sum_erodep_pts = deepcopy(pred_erodep_pts)
		burnsamples = int(samples*self.burn_in)
		pos_param = np.zeros((samples,v_current.size))
		s_pos_param = np.zeros((samples, v_current.size))
		prop_list = np.zeros((samples,v_current.size))
		list_yslicepred = np.zeros((samples,self.real_elev.shape[0])) #slice taken at mid of topography along y axis  
		list_xslicepred = np.zeros((samples,self.real_elev.shape[1])) #slice taken at mid of topography along x axis  
		ymid = int(self.real_elev.shape[1]/2 ) #   cut the slice in the middle 
		xmid = int(self.real_elev.shape[0]/2)
		list_erodep  = np.zeros((samples,pred_erodep_pts[self.simtime].size))
		list_erodep_time  = np.zeros((samples , self.sim_interval.size , pred_erodep_pts[self.simtime].size))
		start = time.time() 
		num_accepted = 0
		num_div = 0 
		surrogate_counter = 0

		with file(('%s/experiment_setting.txt' % (self.filename)),'a') as outfile:
			outfile.write('\nsamples:,{0}'.format(self.samples))
			outfile.write('\nburnin:,{0}'.format(self.burn_in))
			outfile.write('\nnum params:,{0}'.format(self.num_param))
			outfile.write('\ninitial_proposed_vec:,{0}'.format(v_proposal))
			outfile.write('\nstepsize_vec:,{0}'.format(stepsize_vec))  
			outfile.write('\nstep_ratio_vec:,{0}'.format(self.stepratio_vec)) 
			outfile.write('\nswap interval:,{0}'.format(self.swap_interval))
			outfile.write('\nsurrogate interval:,{0}'.format(self.surrogate_interval))
			outfile.write('\nsim interval:,{0}'.format(self.sim_interval))
			outfile.write('\nsurrogate probability:,{0}'.format(self.surrogate_prob))
			outfile.write('\nlikelihood_sed (T/F):,{0}'.format(self.likelihood_sed))
			outfile.write('\nerodep_coords:,{0}'.format(self.erodep_coords))
			outfile.write('\nsed scaling factor:,{0}'.format(self.sedscalingfactor))

		for i in range(samples-1):

			ku = random.uniform(0,1)
			timer1 = time.time()
			is_true_lhood = True
			print 'Sample : ', i, ' Chain :', self.temperature

			# Update by perturbing all the  parameters via "random-walk" sampler and check limits
			v_proposal =  np.random.normal(v_current,stepsize_vec)

			for j in range(v_current.size):
				if v_proposal[j] > self.maxlim_param[j]:
					v_proposal[j] = v_current[j]
				elif v_proposal[j] < self.minlim_param[j]:
					v_proposal[j] = v_current[j]

			# [likelihood_proposal, predicted_elev, pred_erodep_pts, surrogate_var] = self.likelihood_function(v_proposal)

			if ku>(1-self.surrogate_prob) and i>=self.surrogate_interval+1 and i>burnsamples:
				if surrogate_model == None:
					minmax = np.loadtxt(self.filename+'/surrogate/minmax.txt')
					self.minY[0,0] = minmax[0]
					self.maxY[0,0] = minmax[1]
					surrogate_model = Surrogate("krnn",surrogate_X.copy(),surrogate_Y.copy(), self.minlim_param, self.maxlim_param, self.minY, self.maxY, self.filename)
					surrogate_likelihood = surrogate_model.predict(v_proposal.reshape(1,v_proposal.shape[0]),False)
					surrogate_likelihood = surrogate_likelihood *(1.0/self.temperature)
						
				elif self.surrogate_init == 0.0:
					surrogate_likelihood = surrogate_model.predict(v_proposal.reshape(1,v_proposal.shape[0]), False)
					surrogate_likelihood = surrogate_likelihood *(1.0/self.temperature)
				else:
					surrogate_likelihood = surrogate_model.predict(v_proposal.reshape(1,v_proposal.shape[0]), True)
					surrogate_likelihood = surrogate_likelihood *(1.0/self.temperature)

				surg_likeh_list[i+1,0] = likelihood_proposal
				surg_likeh_list[i+1,1] = surrogate_likelihood
				print '\nSample : ', i, ' Chain :', self.temperature, ' -A', likelihood_proposal, ' vs. P ', surrogate_likelihood
				surrogate_counter += 1

				likelihood_proposal = surrogate_likelihood

			else:
				[likelihood_proposal, predicted_elev, pred_erodep_pts, surrogate_var] = self.likelihood_function(v_proposal)
				surg_likeh_list[i+1,0] = likelihood_proposal
				surg_likeh_list[i+1,1] = np.nan

			final_predtopo= predicted_elev[self.simtime]
			pred_erodep = pred_erodep_pts[self.simtime]
 
			# Difference in likelihood from previous accepted proposal
			diff_likelihood = likelihood_proposal - likelihood

			try:
				mh_prob = min(1, math.exp(diff_likelihood))
			except OverflowError as e:
				mh_prob = 1

			u = random.uniform(0,1)

			accept_list[i+1] = num_accepted

			# tausq_list[i+1,0] = tausq
			likeh_list[i+1,0] = surrogate_var
			prop_list[i+1,] = v_proposal

			if u < mh_prob: # Accept sample

				# print 'Sample : ',i , 'Chain ', self.temperature ,' - tausq: ', tausq, ' Accepted' # ,'V_Proposal: ', v_proposal,'Likelihood: ',likelihood_proposal, 'Temp : ',self.temperature,'Accept count: ', num_accepted, ' is accepted')
				# Append sample number to accepted list
				count_list.append(i)            
				likelihood = likelihood_proposal
				v_current = v_proposal
				
				pos_param[i+1,:] = v_current # features rain, erodibility and others  (random walks is only done for this vector)
				likeh_list[i+1,1] = likelihood_proposal				
				#slice taken at mid of topography along y axis  and x axis
				list_yslicepred[i+1,:] =  final_predtopo[:, ymid] 
				list_xslicepred[i+1,:]=   final_predtopo[xmid, :]
				list_erodep[i+1,:] = pred_erodep

				for x in range(self.sim_interval.size): 
					list_erodep_time[i+1,x, :] = pred_erodep_pts[self.sim_interval[x]]

				num_accepted = num_accepted + 1 

				prev_accepted_elev.update(predicted_elev)

				if i>burnsamples and is_true_lhood: 
					
					for k, v in prev_accepted_elev.items():
						sum_elev[k] += v 

					for k, v in pred_erodep_pts.items():
						sum_erodep_pts[k] += v

					num_div += 1

			else: # Reject sample
				# tausq_list[i+1, 1] = tausq_list[i,1]
				likeh_list[i+1, 1] = likeh_list[i,1]
				pos_param[i+1,:] = pos_param[i,:]
				list_yslicepred[i+1,:] =  list_yslicepred[i,:] 
				list_xslicepred[i+1,:]=   list_xslicepred[i,:]
				list_erodep[i+1,:] = list_erodep[i,:]
				list_erodep_time[i+1,:, :] = list_erodep_time[i,:, :]
 
				if i>burnsamples and is_true_lhood:

					# print('saving sum ele')
					for k, v in prev_accepted_elev.items():
						sum_elev[k] += v

					for k, v in prev_acpt_erodep_pts.items():
						sum_erodep_pts[k] += v

					num_div += 1

			if (i % self.swap_interval == 0 ):

				if i> burnsamples and self.runninghisto == True:
					hist, bin_edges = np.histogram(pos_param[burnsamples:i,0], density=True)
					plt.hist(pos_param[burnsamples:i,0], bins='auto')  # arguments are passed to np.histogram
					plt.title("Parameter 1 Histogram")

					file_name = self.filename + '/posterior/pos_parameters/hist_current' + str(self.temperature)
					plt.savefig(file_name+'_0.png')
					plt.clf()

					np.savetxt(file_name+'.txt',  pos_param[ :i,:] ,  fmt='%1.9f')

					hist, bin_edges = np.histogram(pos_param[burnsamples:i,1], density=True)
					plt.hist(pos_param[burnsamples:i,1], bins='auto')  # arguments are passed to np.histogram
					plt.title("Parameter 2 Histogram")
 
					plt.savefig(file_name + '_1.png')
					plt.clf()
 
				other = np.asarray([likelihood])
				temperature = np.asarray([self.temperature])
				samp = np.asarray([i])
				param = np.concatenate([v_current,other,temperature, samp])   
 
				# paramater placed in queue for swapping between chains
				self.parameter_queue.put(param)
				
				#signal main process to start and start waiting for signal for main
				self.signal_main.set()         
				self.event.wait()
				# retrieve parametsrs fom ques if it has been swapped
				if not self.parameter_queue.empty() : 
					try:
						result = self.parameter_queue.get()
 
						v_current= result[0:v_current.size]     
						likelihood = result[v_current.size]
						# self.temperature = result[v_current.size+1]
					except:
						print ('error')

			#SURROGATE TRAINING
			if (i%self.surrogate_interval == 0) and (i!=0):
				#Train the surrogate with the posteriors and likelihood
				surrogate_X, surrogate_Y = prop_list[i+1-self.surrogate_interval:i,:],likeh_list[i+1-self.surrogate_interval:i,0]

				surrogate_Y = surrogate_Y.reshape(surrogate_Y.shape[0],1)
				param = np.concatenate([surrogate_X, surrogate_Y],axis=1)
				self.surrogate_parameterqueue.put(param)
				self.surrogate_start.set()
				self.surrogate_resume.wait()
				
				model_sign = np.loadtxt(self.filename+'/surrogate/model_signature.txt')
				self.model_signature = model_sign 
				
				if self.model_signature==1.0:
					minmax = np.loadtxt(self.filename+'/surrogate/minmax.txt')
					self.minY[0,0] = minmax[0]
					self.maxY[0,0] = minmax[1]
					# print 'min ', self.minY, ' max ', self.maxY
					dummy_X = np.zeros((1,1))
					dummy_Y = np.zeros((1,1))
					surrogate_model = Surrogate("krnn", dummy_X, dummy_Y, self.minlim_param, self.maxlim_param, self.minY, self.maxY, self.filename)
				
				self.surrogate_init = surrogate_model.predict(v_proposal.reshape(1,v_proposal.shape[0]), False)				
				print "Surrogate init ", self.surrogate_init , " - should be -1"

		accepted_count =  len(count_list) 
		accept_ratio = accepted_count / (samples * 1.0) * 100
		#--------------------------------------------------------------- 

		other = np.asarray([likelihood])
		temperature = np.asarray([self.temperature])
		samp = np.asarray([i])
		param = np.concatenate([v_current,other,temperature, samp])   

		self.parameter_queue.put(param)
 
		file_name = self.filename+'/posterior/pos_parameters/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,pos_param ) 

		file_name = self.filename+'/posterior/pos_parameters/prop_list/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, prop_list )

		file_name = self.filename+'/posterior/predicted_topo/chain_xslice_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, list_xslicepred )

		file_name = self.filename+'/posterior/predicted_topo/chain_yslice_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, list_yslicepred )

		file_name = self.filename+'/posterior/pos_likelihood/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,likeh_list, fmt='%1.4f') 
		
		file_name = self.filename+'/posterior/surg_likelihood/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,surg_likeh_list, fmt='%1.4f')

		file_name = self.filename + '/posterior/accept_list/chain_' + str(self.temperature) + '_accept.txt'
		np.savetxt(file_name, [accept_ratio], fmt='%1.4f')

		file_name = self.filename + '/posterior/accept_list/chain_' + str(self.temperature) + '.txt'
		np.savetxt(file_name, accept_list, fmt='%1.4f')
 
 		# print('Amount of samples added in topography dict: ', self.temperature,' : ', num_div)
 		print('surrogate counter for chain ', self.temperature, ' : ', surrogate_counter)

		for s in range(self.sim_interval.size):  
			file_name = self.filename + '/posterior/predicted_erodep/chain_' + str(self.sim_interval[s]) + '_' + str(self.temperature) + '.txt'
			np.savetxt(file_name, list_erodep_time[:,s, :] , fmt='%.2f') 

		for k, v in sum_elev.items():
			sum_elev[k] = np.divide(sum_elev[k], num_div)
			mean_pred_elevation = sum_elev[k]

			sum_erodep_pts[k] = np.divide(sum_erodep_pts[k], num_div)
			mean_pred_erodep_pnts = sum_erodep_pts[k]

			file_name = self.filename + '/posterior/predicted_topo/chain_' + str(k) + '_' + str(self.temperature) + '.txt'
			np.savetxt(file_name, mean_pred_elevation, fmt='%.2f')
 		
		self.signal_main.set()
		self.surrogate_start.set()
		
		return 

class ParallelTempering:
	def __init__(self, vec_param, num_chains, maxtemp, samples ,swap_interval, filename, realvalues_vec,
		num_param,  real_elev, erodep_pts, erodep_coords, simtime, sim_interval, resolu_factor,
		run_nb, inputxml, surrogate_interval, surrogate_prob):
		
		#Parallel Tempering variables
		self.swap_interval = swap_interval
		self.filename = filename
		self.maxtemp = maxtemp
		self.num_swap = 0
		self.total_swap_proposals = 0
		self.num_chains = num_chains
		self.chains = []
		self.temperatures = []
		self.samples = int(samples/self.num_chains)
		self.sub_sample_size = max(1, int(0.05* self.samples))
		self.show_fulluncertainity = False # needed in cases when you reall want to see full prediction of 5th and 95th percentile of topo. takes more space 
		self.real_erodep_pts  = erodep_pts
		self.real_elev = real_elev
		self.resolu_factor =  resolu_factor
		self.num_param = num_param
		self.erodep_coords = erodep_coords
		self.simtime = simtime
		self.sim_interval = sim_interval
		self.run_nb =run_nb 
		self.xmlinput = inputxml
		self.vec_param = vec_param
		self.realvalues  =  realvalues_vec 
		# create queues for transfer of parameters between process chain
		self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
		self.chain_queue = multiprocessing.JoinableQueue()	
		self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]
		self.event = [multiprocessing.Event() for i in range (self.num_chains)]
		# create variables for surrogates
		self.model_signature = 0.0
		self.surrogate_interval = surrogate_interval
		self.surrogate_prob = surrogate_prob
		self.surrogate_resume_events = [multiprocessing.Event() for i in range(self.num_chains)]
		self.surrogate_start_events = [multiprocessing.Event() for i in range(self.num_chains)]
		self.surrogate_parameterqueues = [multiprocessing.Queue() for i in range(self.num_chains)]
		self.surrchain_queue = multiprocessing.JoinableQueue()
		self.all_param = None
		self.minlim_param = 0.0
		self.maxlim_param = 0.0
		self.minY = np.zeros((1,1))
		self.maxY = np.ones((1,1))

	def assign_temperatures(self):
		tmpr_rate = (self.maxtemp /self.num_chains)
		temp = 2.0
		for i in xrange(0, self.num_chains):
			self.temperatures.append(temp)
			temp += tmpr_rate
			print(self.temperatures[i])

	def initialize_chains (self, minlim_param, maxlim_param, stepratio_vec, likelihood_sed, burn_in):
		self.burn_in = burn_in
		self.minlim_param = minlim_param
		self.maxlim_param = maxlim_param

		self.vec_param =   np.random.uniform(minlim_param, maxlim_param) # will begin from diff position in each replica (comment if not needed)
		self.assign_temperatures()
		for i in xrange(0, self.num_chains):
			self.chains.append(PtReplica(self.num_param, self.vec_param, minlim_param, maxlim_param,
			stepratio_vec, likelihood_sed, self.swap_interval, self.sim_interval, self.simtime, 
			self.samples, self.real_elev, self.real_erodep_pts, self.erodep_coords, self.filename, 
			self.xmlinput, self.run_nb, self.temperatures[i], self.burn_in, self.parameter_queue[i], 
			self.event[i], self.wait_chain[i],self.surrogate_parameterqueues[i],self.surrogate_interval,
			self.surrogate_prob,self.surrogate_start_events[i],self.surrogate_resume_events[i]))

	def surr_procedure(self,queue):

		if queue.empty() is False:
			return queue.get()
		else:
			return

	def swap_procedure(self, parameter_queue_1, parameter_queue_2):
		if parameter_queue_2.empty() is False and parameter_queue_1.empty() is False:
			param1 = parameter_queue_1.get()
			param2 = parameter_queue_2.get()
			
			v1 = param1[0:self.num_param]
			lhood1 = param1[self.num_param]
			T1 = param1[self.num_param+1]

			v2 = param2[0:self.num_param]
			lhood2 = param2[self.num_param]
			T2 = param2[self.num_param+1]
			#print('yo')
			#SWAPPING PROBABILITIES
			try:
				swap_proposal = min(1, 0.5*math.exp(lhood1-lhood2))
			except OverflowError as e:
				swap_proposal = 1

			# swap_proposal =  (lhood1/[1 if lhood2 == 0 else lhood2])*(1/T1 * 1/T2)
			u = np.random.uniform(0,1)
			if u < swap_proposal:
				# print('SWAPPED')
				self.num_swap += 1
				param_temp =  param1
				param1 = param2
				param2 = param_temp
			return param1, param2
		else:
			return

	def surrogate_trainer(self,params):

		X = params[:,:self.num_param]
		Y = params[:,self.num_param].reshape(X.shape[0],1)
		
		for i in range(Y.shape[1]):
			min_Y = min(Y[:,i])
			max_Y = max(Y[:,i])
			self.minY[0,i] = min_Y
			self.maxY[0,i] = max_Y 

		self.model_signature += 1.0
		
		if self.model_signature == 1.0:
			np.savetxt(self.filename+'/surrogate/minmax.txt',[self.minY, self.maxY])
		
		np.savetxt(self.filename+'/surrogate/model_signature.txt', [self.model_signature])
		
		Y= self.normalize_likelihood(Y)
		indices = np.where(Y==np.inf)[0]
		X = np.delete(X, indices, axis=0)
		Y = np.delete(Y,indices, axis=0)
		surrogate_model = Surrogate("krnn", X , Y , self.minlim_param, self.maxlim_param, self.minY, self.maxY, self.filename)
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

	def SurrogateHeatmap(self, filename, x1Data, x2Data, yData, y_min, y_max, title):
		trace = go.Heatmap(x=x1Data, y=x2Data, z=yData, zmin = y_min, zmax = y_max)
		data = [trace]
		layout = Layout(
			title='%s ' %(title),
			scene=Scene(
				zaxis=ZAxis(title = 'Likl',range=[-1,0] ,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				xaxis=XAxis(title = 'Rain', range=[x2Data.min(),x2Data.max()],gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				yaxis=YAxis(title = 'Erod', range=[x2Data.min(),x2Data.max()] ,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				bgcolor="rgb(244, 244, 248)"
			)
		)
		fig = Figure(data=data, layout=layout)
		plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/%s.html' %(filename, title), validate=False)
		return

	def run_chains(self):
		# only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
		swap_proposal = np.ones(self.num_chains-1) 
		
		# create parameter holders for paramaters that will be swapped
		replica_param = np.zeros((self.num_chains, self.num_param))  
		lhood = np.zeros(self.num_chains)

		# Define the starting and ending of MCMC Chains
		start = 0
		end = self.samples-1
		number_exchange = np.zeros(self.num_chains)

		#-------------------------------------------------------------------------------------
		# run the MCMC chains
		#-------------------------------------------------------------------------------------
		for i in range(0,self.num_chains):
			self.chains[i].start_chain = start
			self.chains[i].end = end
		
		#-------------------------------------------------------------------------------------
		# run the MCMC chains
		#-------------------------------------------------------------------------------------
		for j in range(0,self.num_chains):        
			self.chains[j].start()

		while True:
			#-------------------------------------------------------------------------------------
			# wait for chains to complete one pass through the samples
			#-------------------------------------------------------------------------------------
			for k in range(0,self.num_chains):
				self.wait_chain[k].wait()
				# print(k)

			#-------------------------------------------------------------------------------------
			#get info from chains and swaps
			#-------------------------------------------------------------------------------------
			for m in range(0,self.num_chains-1):
				#print('starting swap')
				self.chain_queue.put(self.swap_procedure(self.parameter_queue[m],self.parameter_queue[m+1])) 
				while True:
					if self.chain_queue.empty():
						self.chain_queue.task_done()
						# print(k,'EMPTY QUEUE')
						break
					swap_process = self.chain_queue.get()
					# print(swap_process)
					if swap_process is None:
						self.chain_queue.task_done()
						# print(k,'No Process')
						break
					param1, param2 = swap_process
					#self.chain_queue.task_done()
					self.parameter_queue[m].put(param1)
					self.parameter_queue[m+1].put(param2)

			#-------------------------------------------------------------------------------------
			# resume suspended process
			#-------------------------------------------------------------------------------------
			for n in range (self.num_chains):
				#print(k)
				self.event[n].set()
			count = 0

			#-------------------------------------------------------------------------------------
			# # Surrogate's Events:
			#-------------------------------------------------------------------------------------
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
					print "Before surrogate trainer set"
					for k in range(self.num_chains):
						self.surrogate_resume_events[k].set()
			
			######################

			#-------------------------------------------------------------------------------------
			#check if all chains have completed runing
			#-------------------------------------------------------------------------------------
			for i in range(self.num_chains):
				if self.chains[i].is_alive() is False:
					count+=1
			if count == self.num_chains  :
				#print(count)
				break

		#-------------------------------------------------------------------------------------
		#wait for all processes to jin the main process
		#-------------------------------------------------------------------------------------    
		#JOIN THEM TO MAIN PROCESS
		for j in range(0,self.num_chains):
			self.chains[j].join()
		self.chain_queue.join()

		pos_param, likelihood_rep, accept_list, pred_topo,  combined_erodep, accept, pred_topofinal, list_xslice, list_yslice = self.show_results('chain_')
		self.view_crosssection_uncertainity(list_xslice, list_yslice)

		optimal_para, para_5thperc, para_95thperc = self.get_uncertainity(likelihood_rep, pos_param)
		np.savetxt(self.filename+'/optimal_percentile_para.txt', [optimal_para, para_5thperc, para_95thperc] )
 
		for s in range(self.num_param):  
			self.plot_figure(pos_param[s,:], 'pos_distri_'+str(s), self.realvalues[s,:]  ) 
 
		for i in range(self.sim_interval.size):
			self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_topo[i,:,:], title='Predicted Topography ', time_frame=self.sim_interval[i],  filename= 'mean')
 
		if self.show_fulluncertainity == True: # this to be used when you need output of the topo predictions - 5th and 95th percentiles
 
			pred_elev5th, pred_eroddep5th, pred_erd_pts5th = self.run_badlands(np.asarray(para_5thperc)) 
			self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_elev5th[self.simtime], title='Pred. Topo. - 5th Percentile', time_frame= self.simtime, filename= '5th')
 
			pred_elev95th, pred_eroddep95th, pred_erd_pts95th = self.run_badlands(para_95thperc)
			self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_elev95th[self.simtime], title='Pred. Topo. - 95th Percentile', time_frame= self.simtime, filename = '95th')

			pred_elevoptimal, pred_eroddepoptimal, pred_erd_optimal = self.run_badlands(optimal_para)
			self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_elevoptimal[self.simtime], title='Pred. Topo. - Optimal', time_frame= self.simtime, filename = 'optimal')

			self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=  self.real_elev , title='Ground truth Topography', time_frame= self.simtime, filename = 'ground_truth')

		print('Number of Swaps = ', self.num_swap, 'process ended')
		return (pos_param,likelihood_rep, accept_list, combined_erodep,pred_topofinal)

	def view_crosssection_uncertainity(self, list_xslice, list_yslice):
		ymid = int(self.real_elev.shape[1]/2 ) #   cut the slice in the middle 
		xmid = int(self.real_elev.shape[0]/2)
		x_ymid_real = self.real_elev[xmid, :] 
		y_xmid_real = self.real_elev[:, ymid ] 
		x_ymid_mean = list_xslice.mean(axis=1)
		x_ymid_5th = np.percentile(list_xslice, 5, axis=1)
		x_ymid_95th= np.percentile(list_xslice, 95, axis=1)
		y_xmid_mean = list_yslice.mean(axis=1)
		y_xmid_5th = np.percentile(list_yslice, 5, axis=1)
		y_xmid_95th= np.percentile(list_yslice, 95, axis=1)
		x = np.linspace(0, x_ymid_mean.size * self.resolu_factor, num=x_ymid_mean.size) 
		x_ = np.linspace(0, y_xmid_mean.size * self.resolu_factor, num=y_xmid_mean.size)
		plt.plot(x, x_ymid_real, label='ground truth') 
		plt.plot(x, x_ymid_mean, label='pred. (mean)')
		plt.plot(x, x_ymid_5th, label='pred.(5th percen.)')
		plt.plot(x, x_ymid_95th, label='pred.(95th percen.)')

		plt.fill_between(x, x_ymid_5th , x_ymid_95th, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right') 

		plt.title("Uncertainty in topography prediction (cross section)  ")
		plt.xlabel(' Distance (km)  ')
		plt.ylabel(' Height in meters')
		
		plt.savefig(self.filename+'/x_ymid_opt.png')  
		plt.savefig(self.filename+'/x_ymid_opt.svg', format='svg', dpi=400)
		plt.clf()

		plt.plot(x_, y_xmid_real, label='ground truth') 
		plt.plot(x_, y_xmid_mean, label='pred. (mean)') 
		plt.plot(x_, y_xmid_5th, label='pred.(5th percen.)')
		plt.plot(x_, y_xmid_95th, label='pred.(95th percen.)')
		plt.xlabel(' Distance (km) ')
		plt.ylabel(' Height in meters')
		
		plt.fill_between(x_, y_xmid_5th , y_xmid_95th, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Uncertainty in topography prediction  (cross section)  ")
		plt.savefig(self.filename+'/y_xmid_opt.png')  
		plt.savefig(self.filename+'/y_xmid_opt.svg', format='svg', dpi=400)

		plt.clf()
	
	def show_results(self, filename):

		burnin = int(self.samples * self.burn_in)
		pos_param = np.zeros((self.num_chains, self.samples - burnin, self.num_param))
		list_xslice = np.zeros((self.num_chains, self.samples - burnin, self.real_elev.shape[1]))
		list_yslice = np.zeros((self.num_chains, self.samples - burnin, self.real_elev.shape[0]))
		likelihood_rep = np.zeros((self.num_chains, self.samples - burnin, 2)) # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
		surg_likelihood = np.zeros((self.num_chains, self.samples - burnin, 2)) # index 1 for likelihood proposal and for gp_prediction
		accept_percent = np.zeros((self.num_chains, 1))
		accept_list = np.zeros((self.num_chains, self.samples )) 
		topo  = self.real_elev
		replica_topo = np.zeros((self.sim_interval.size, self.num_chains, topo.shape[0], topo.shape[1])) #3D
		combined_topo = np.zeros((self.sim_interval.size, topo.shape[0], topo.shape[1]))
		replica_erodep_pts = np.zeros((self.num_chains, self.real_erodep_pts.shape[1] )) 
		combined_erodep = np.zeros((self.sim_interval.size, self.num_chains, self.samples - burnin, self.real_erodep_pts.shape[1] ))
		timespan_erodep = np.zeros((self.sim_interval.size,  (self.samples - burnin) * self.num_chains, self.real_erodep_pts.shape[1] ))
 
		for i in range(self.num_chains):
			file_name = self.filename + '/posterior/pos_parameters/'+filename + str(self.temperatures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			pos_param[i, :, :] = dat[burnin:,:]
  
			file_name = self.filename + '/posterior/predicted_topo/chain_xslice_'+  str(self.temperatures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			list_xslice[i, :, :] = dat[burnin:,:] 

			file_name = self.filename + '/posterior/predicted_topo/chain_yslice_'+  str(self.temperatures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			list_yslice[i, :, :] = dat[burnin:,:] 

			file_name = self.filename + '/posterior/pos_likelihood/'+filename + str(self.temperatures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			likelihood_rep[i, :] = dat[burnin:]

			file_name = self.filename + '/posterior/surg_likelihood/'+filename + str(self.temperatures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			surg_likelihood[i, :] = dat[burnin:]

			file_name = self.filename + '/posterior/accept_list/' + filename + str(self.temperatures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			accept_list[i, :] = dat 

			file_name = self.filename + '/posterior/accept_list/' + filename + str(self.temperatures[i]) + '_accept.txt'
			dat = np.loadtxt(file_name) 
			accept_percent[i, :] = dat

			for j in range(self.sim_interval.size):

				file_name = self.filename+'/posterior/predicted_topo/chain_'+str(self.sim_interval[j])+'_'+ str(self.temperatures[i])+ '.txt'
				dat_topo = np.loadtxt(file_name)
				replica_topo[j,i,:,:] = dat_topo

				file_name = self.filename+'/posterior/predicted_erodep/chain_'+str(self.sim_interval[j])+'_'+ str(self.temperatures[i])+ '.txt'
				dat_erodep = np.loadtxt(file_name)
				combined_erodep[j,i,:,:] = dat_erodep[burnin:,:]


		posterior = pos_param.transpose(2,0,1).reshape(self.num_param,-1)    
		xslice = list_xslice.transpose(2,0,1).reshape(self.real_elev.shape[1],-1) 
		yslice = list_yslice.transpose(2,0,1).reshape(self.real_elev.shape[0],-1)
		likelihood_vec = likelihood_rep.transpose(2,0,1).reshape(2,-1) 
		surg_likelihood_vec = surg_likelihood.transpose(2,0,1).reshape(2,-1)

		for j in range(self.sim_interval.size):
			for i in range(self.num_chains):
				combined_topo[j,:,:] += replica_topo[j,i,:,:]  
			combined_topo[j,:,:] = combined_topo[j,:,:]/self.num_chains

			dx = combined_erodep[j,:,:,:].transpose(2,0,1).reshape(self.real_erodep_pts.shape[1],-1)

			timespan_erodep[j,:,:] = dx.T

		surrogate_likl = surg_likelihood_vec.T
		surrogate_likl = surrogate_likl[~np.isnan(surrogate_likl).any(axis=1)]

		slen = np.arange(0,surrogate_likl.shape[0],1)
		fig = plt.figure(figsize = (12,12))
		ax = fig.add_subplot(111)
		ax.set_facecolor('#f2f2f3')
		surrogate_plot = ax.plot(slen,surrogate_likl[:,1],linestyle='-', linewidth= 1, color= 'b', label= 'Surrogate Likelihood')
		model_plot = ax.plot(slen,surrogate_likl[:,0],linestyle= '--', linewidth = 1, color = 'k', label = 'Model Likelihood')
		ax.set_title('Surrogate predicted likelihood',size= 9+2)
		ax.set_xlabel('Samples',size= 9+1)
		ax.set_ylabel('Tau/Likelihood', size= 9+1)
		ax.set_xlim([0,np.amax(slen)])
		ax.legend((surrogate_plot, model_plot),('Surrogate Likelihood', 'Model Likelihood'))
		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		plt.savefig('%s/surrogate_likl.png'% (self.filename), dpi=300, transparent=False)
		plt.clf()

		accept = np.sum(accept_percent)/self.num_chains
 
		pred_topofinal = combined_topo[-1,:,:] # get the last mean pedicted topo to calculate mean squared error loss 


		# creating test data set for testing with SGD-FNN verification
		y_norm = likelihood_vec.T[:,0]
		y_norm = y_norm.reshape(y_norm.shape[0],1)
		Y_norm = self.normalize_likelihood(y_norm)

		np.savetxt(self.filename + '/surrogate/prediction_benchmark_data/norm_likelihood.txt', Y_norm)

		np.savetxt(self.filename + '/pos_param.txt', posterior.T) 
		
		np.savetxt(self.filename + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')

		np.savetxt(self.filename + '/surrogate/surg_likelihood.txt', surrogate_likl, fmt='%1.5f')

		np.savetxt(self.filename + '/accept_list.txt', accept_list, fmt='%1.2f')
  
		np.savetxt(self.filename + '/acceptpercent.txt', [accept], fmt='%1.2f')

		return posterior, likelihood_vec.T, accept_list, combined_topo, timespan_erodep, accept, pred_topofinal, xslice, yslice # Merge different MCMC chains y stacking them on top of each other
	
	def find_nearest(self, array,value): # just to find nearest value of a percentile (5th or 9th from pos likelihood)
		idx = (np.abs(array-value)).argmin()
		return array[idx], idx

	def get_uncertainity(self, likelihood_rep, pos_param ): 

		likelihood_pos = likelihood_rep[:,1]
		a = np.percentile(likelihood_pos, 5)   
		lhood_5thpercentile, index_5th = self.find_nearest(likelihood_pos,a)  
		b = np.percentile(likelihood_pos, 95) 
		lhood_95thpercentile, index_95th = self.find_nearest(likelihood_pos,b)  
		max_index = np.argmax(likelihood_pos) # find max of pos liklihood to get the max or optimal pos value  
		optimal_para = pos_param[:, max_index] 
		para_5thperc = pos_param[:, index_5th]
		para_95thperc = pos_param[:, index_95th] 

		return optimal_para, para_5thperc, para_95thperc

	def run_badlands(self, input_vector): # this is same method in Replica class - copied here to get error uncertainity in topo pred

		model = badlandsModel()

		# Load the XmL input file
		model.load_xml(str(self.run_nb), self.xmlinput, muted=True)

		# print(input_vector, ' input badlands')

		# Adjust erodibility based on given parameter
		model.input.SPLero = input_vector[1] 
		model.flow.erodibility.fill(input_vector[1] )

		# Adjust precipitation values based on given parameter
		model.force.rainVal[:] = input_vector[0] 

		# Adjust m and n values
		model.input.SPLm = input_vector[2] 
		model.input.SPLn = input_vector[3] 

		elev_vec = collections.OrderedDict()
		erodep_vec = collections.OrderedDict()
		erodep_pts_vec = collections.OrderedDict()

		for x in range(len(self.sim_interval)):
			self.simtime = self.sim_interval[x]

			model.run_to_time(self.simtime, muted=True)
		 
			elev, erodep = self.interpolate_array(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)

			erodep_pts = np.zeros((self.erodep_coords.shape[0]))

			for count, val in enumerate(self.erodep_coords):
				erodep_pts[count] = erodep[val[0], val[1]]

			elev_vec[self.simtime] = elev
			erodep_vec[self.simtime] = erodep

			erodep_pts_vec[self.simtime] = erodep_pts

		return elev_vec, erodep_vec, erodep_pts_vec

	def interpolate_array(self, coords=None, z=None, dz=None):
		"""
		Interpolate the irregular spaced dataset from badlands on a regular grid.
		"""
		x, y = np.hsplit(coords, 2)
		dx = (x[1]-x[0])[0]

		nx = int((x.max() - x.min())/dx+1)
		ny = int((y.max() - y.min())/dx+1)
		xi = np.linspace(x.min(), x.max(), nx)
		yi = np.linspace(y.min(), y.max(), ny)

		xi, yi = np.meshgrid(xi, yi)
		xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
		XY = np.column_stack((x,y))

		tree = cKDTree(XY)
		distances, indices = tree.query(xyi, k=3)
		if len(z[indices].shape) == 3:
			z_vals = z[indices][:,:,0]
			dz_vals = dz[indices][:,:,0]
		else:
			z_vals = z[indices]
			dz_vals = dz[indices]

		zi = np.average(z_vals,weights=(1./distances), axis=1)
		dzi = np.average(dz_vals,weights=(1./distances), axis=1)
		onIDs = np.where(distances[:,0] == 0)[0]
		if len(onIDs) > 0:
			zi[onIDs] = z[indices[onIDs,0]]
			dzi[onIDs] = dz[indices[onIDs,0]]
		zreg = np.reshape(zi,(ny,nx))
		dzreg = np.reshape(dzi,(ny,nx))
		return zreg,dzreg

	def plot_figure(self, list, title, real_value ): 

		list_points = list
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
		for count, v in enumerate(real_value):
			ax1.axvline(x=v, color='%s' %(color[count]), linestyle='dashed', linewidth=1) # comment when go real value is 

		ax1.grid(True)
		ax1.set_ylabel('Frequency',size= font+1)
		ax1.set_xlabel('Parameter values', size= font+1)
	
		ax2 = fig.add_subplot(212)
		list_points = np.asarray(np.split(list_points, self.num_chains))
		ax2.set_facecolor('#f2f2f3') 
		ax2.plot(list_points.T , label=None)
		ax2.set_title(r'Trace plot',size= font+2)
		ax2.set_xlabel('Samples',size= font+1)
		ax2.set_ylabel('Parameter values', size= font+1) 

		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		plt.savefig(self.filename + '/' + title  + '_pos_.png', bbox_inches='tight', dpi=300, transparent=False)
		plt.clf()

	def viewGrid(self, width=1000, height=1000, zmin=None, zmax=None, zData=None, title='Predicted Topography', time_frame=None, filename=None):

		if zmin == None:
			zmin =  zData.min()

		if zmax == None:
			zmax =  zData.max()

		tickvals= [0,50,75,-50]

		xx = (np.linspace(0, zData.shape[0]* self.resolu_factor, num=zData.shape[0]/10 )) 
		yy = (np.linspace(0, zData.shape[1] * self.resolu_factor, num=zData.shape[1]/10 )) 
		xx = np.around(xx, decimals=0)
		yy = np.around(yy, decimals=0)

		data = Data([Surface(x= zData.shape[0] , y= zData.shape[1] , z=zData, colorscale='YIGnBu')])
		#if self.problem ==1 or self.problem == 2: # quick fix just when you want to downcale the axis as in crater problems 

		layout = Layout(title='Predicted Topography' , autosize=True, width=width, height=height,scene=Scene(
					zaxis=ZAxis(title = 'Elevation (m)   ', range=[zmin, zmax], autorange=False, nticks=6, gridcolor='rgb(255, 255, 255)',
								gridwidth=2, zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2),
					xaxis=XAxis(title = 'x-coordinates  ',  tickvals= xx,      gridcolor='rgb(255, 255, 255)', gridwidth=2,
								zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2),
					yaxis=YAxis(title = 'y-coordinates  ', tickvals= yy,    gridcolor='rgb(255, 255, 255)', gridwidth=2,
								zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2),
					bgcolor="rgb(244, 244, 248)"
				)
			)

		fig = Figure(data=data, layout=layout) 
		graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename= self.filename +  '/pred_plots'+ '/pred_'+filename+'_'+str(time_frame)+ '_.html', validate=False)

		elev_data = np.reshape(zData, zData.shape[0] * zData.shape[1] )   
		hist, bin_edges = np.histogram(elev_data, density=True)
		plt.hist(elev_data, bins='auto')  
		plt.title("Predicted Topography Histogram")  
		plt.xlabel('Height in meters')
		plt.ylabel('Frequency')
		plt.savefig(self.filename + '/pred_plots'+'/pred_'+filename+'_'+str(time_frame)+ '_.png')
		plt.clf()

	def mean_sq_error(self, pred_erodep, pred_elev,  real_elev,  real_erodep_pts): 
		elev = np.sqrt(np.sum(np.square(pred_elev -  real_elev))  / real_elev.size)  
		sed =  np.sqrt(np.sum(np.square(pred_erodep -  real_erodep_pts)) / real_erodep_pts.size  ) 

		return elev + sed, sed

	def make_directory (self, directory): 
		if not os.path.exists(directory):
			os.makedirs(directory)

	def plot_erodep(self, erodep_mean, erodep_std, groundtruth_erodep_pts, sim_interval, filename):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		index = np.arange(groundtruth_erodep_pts.size) 
		ground_erodepstd = np.zeros(groundtruth_erodep_pts.size) 
		opacity = 0.8
		width = 0.35                      # the width of the bars

		## the bars
		rects1 = ax.bar(index, erodep_mean, width,
					color='blue',
					yerr=erodep_std,
					error_kw=dict(elinewidth=2,ecolor='red'))

		rects2 = ax.bar(index+width, groundtruth_erodep_pts, width, color='green', 
					yerr=ground_erodepstd,
					error_kw=dict(elinewidth=2,ecolor='red') )

		ax.set_ylabel('Height in meters')
		ax.set_xlabel('Selected Coordinates')
		ax.set_title('Erosion Deposition')

		xTickMarks = [str(i) for i in range(1,21)]
		ax.set_xticks(index+width)
		xtickNames = ax.set_xticklabels(xTickMarks)
		plt.setp(xtickNames, rotation=0, fontsize=8)

		## add a legend
		plotlegend = ax.legend((rects1[0], rects2[0]), ('Predicted  ', ' Ground-truth ') )
		 
		plt.savefig(filename +'/pred_plots/pos_erodep_'+str(sim_interval) +'_.png')
		plt.clf() 
		return  

	def plot_boxplot(self, pos_param, filename):
		mpl_fig = plt.figure()
		ax = mpl_fig.add_subplot(111)
		ax.boxplot(pos_param.T) 
		ax.set_xlabel('Badlands parameters')
		ax.set_ylabel('Posterior') 
		plt.legend(loc='upper right') 
		plt.title("Boxplot of Posterior")
		plt.savefig(filename+'/pred_plots/badlands_pos.png')
		plt.savefig(filename+'/pred_plots/badlands_pos.svg', format='svg', dpi=400)

def main():
	
	random.seed(time.time()) 
	samples = 2000 #total number of samples by all the chains (replicas) in parallel tempering
	problem = 3 # input("Which problem do you want to choose 1- Crater 2- Etopo")
	
	#-------------------------------------------------------------------------------------
	# Number of chains of MCMC required to be run. PT is a multicore implementation must num_chains >= 2
	# Choose a value less than the numbe of core available (avoid context swtiching)
	#-------------------------------------------------------------------------------------
	num_chains = 4
	swap_ratio = 0.1    #adapt these 
	surr_ratio = 0.1
	burn_in =0.1
	maxtemp = int(num_chains * 5)/2
	
	surrogate_assist = True
	surrogate_prob = 0.1
	
	swap_interval = int(swap_ratio*(samples/num_chains)) #how ofen you swap neighbours
	surrogate_interval = int(surr_ratio*(samples/num_chains))
	burn_sampl_chain = int(samples*burn_in/num_chains)
	
	if surrogate_interval < swap_interval:
		surrogate_interval = swap_interval
	if surrogate_interval%swap_interval!=0:
		surrogate_interval = surrogate_interval+swap_interval - surrogate_interval%swap_interval

	print 'swap interval ',swap_interval,' \nsurrogate interval ',surrogate_interval,'\nburn samples/chain ',burn_sampl_chain 
	
	if problem == 1:
		problemfolder = 'Examples/crater/'
		xmlinput = problemfolder + 'crater.xml'
		# print('xmlinput', xmlinput)
		simtime = 50000
		resolu_factor = 0.002 # this helps visualize the surface distance in meters 
		true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
		m = 0.5 # used to be constants  
		n = 1
		real_rain = 1.5
		real_erod = 5.e-5  
		likelihood_sediment = True
		maxlim_param = [3.0, 7.e-5, 2.0, 2.0]  # [rain, erod] this can be made into larger vector, with region based rainfall, or addition of other parameters
		minlim_param = [0.0, 3.e-5, 0.0, 0.0]   # hence, for 4 regions of rain and erod[rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod_reg1, erod_reg2, erod_reg3, erod_reg4 ]
									# hence, for 4 regions of rain and 1 erod, plus other free parameters (p1, p2) [rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod, p1, p2 ]
									#if you want to freeze a parameter, keep max and min limits the same
		vec_param = np.random.uniform(minlim_param, maxlim_param) #  draw intial values for each of the free parameters
		stepsize_ratio  = 0.02 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now
		stepratio_vec =  np.repeat(stepsize_ratio, vec_param.size) 
		num_param = vec_param.size 
		erodep_coords =  np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]]) # need to hand pick given your problem

		if (true_parameter_vec.shape[0] != vec_param.size ) :
			print('seems your true parameters file in data folder is not updated with true values of your parameters.  ')
			print('make sure that this is updated in case when you intro more parameters. should have as many rows as parameters ')
			return

	elif problem == 2:
		problemfolder = 'Examples/etopo/'
		xmlinput = problemfolder + 'etopo.xml'
		simtime = 1000000
		resolu_factor = 1000
		true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
		m = 0.5 # used to be constants  
		n = 1
		real_rain = 1.5
		real_erod = 5.e-5
		likelihood_sediment = True
		real_caerial = 8.e-1 
		real_cmarine = 5.e-1 # Marine diffusion coefficient [m2/a] -->
		maxlim_param = [3.0,7.e-6, 2, 2,  1.0, 0.7]  # [rain, erod] this can be made into larger vector, with region based rainfall, or addition of other parameters
		minlim_param = [0.0 ,3.e-6, 0, 0, 0.6, 0.3 ]   # hence, for 4 regions of rain and erod[rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod_reg1, erod_reg2, erod_reg3, erod_reg4 ]
									## hence, for 4 regions of rain and 1 erod, plus other free parameters (p1, p2) [rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod, p1, p2 ]
									#if you want to freeze a parameter, keep max and min limits the same
		vec_param = np.random.uniform(minlim_param, maxlim_param) #  draw intial values for each of the free parameters
		stepsize_ratio  = 0.02 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now
		stepratio_vec =  np.repeat(stepsize_ratio, vec_param.size) 
		num_param = vec_param.size
		# print(vec_param) 
		erodep_coords = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]])  # need to hand pick given your problem

		if (true_parameter_vec.shape[0] != vec_param.size ): 
			print(' seems your true parameters file in data folder is not updated with true values of your parameters.  ')
			print('make sure that this is updated in case when you intro more parameters. should have as many rows as parameters ') 
			return
	
	elif problem == 3:
		problemfolder = 'Examples/etopo_fast/'
		xmlinput = problemfolder + 'etopo.xml'
		simtime = 500000
		resolu_factor = 1.5
		true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt') # make sure that this is updated in case when you intro more parameters. should have as many rows as parameters

		m = 0.5 # used to be constants  
		n = 1
		real_rain = 1.5
		real_erod = 5.e-6
		real_caerial = 8.e-1 
		real_cmarine = 5.e-1 # Marine diffusion coefficient [m2/a] -->
		likelihood_sediment = False

		maxlim_param = [3.0,7.e-6, 2, 2, 1.0, 0.7]  # [rain, erod] this can be made into larger vector, with region based rainfall, or addition of other parameters
		minlim_param = [0.0 ,3.e-6, 0, 0, 0.6, 0.3 ]   # hence, for 4 regions of rain and erod[rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod_reg1, erod_reg2, erod_reg3, erod_reg4 ]
									## hence, for 4 regions of rain and 1 erod, plus other free parameters (p1, p2) [rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod, p1, p2 ]
									#if you want to freeze a parameter, keep max and min limits the same
		vec_param = np.random.uniform(minlim_param, maxlim_param) #  draw intial values for each of the free parameters
		stepsize_ratio  = 0.02 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now
		stepratio_vec =  np.repeat(stepsize_ratio, vec_param.size) #[stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio]
		num_param = vec_param.size
		erodep_coords =  np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]]) # need to hand pick given your problem

		if (true_parameter_vec.shape[0] != vec_param.size ):
			print(' seems your true parameters file in data folder is not updated with true values of your parameters.  ')
			print('make sure that this is updated in case when you intro more parameters. should have as many rows as parameters ') 
			return 

	elif problem == 4:
		problemfolder = 'Examples/mountain/'
		xmlinput = problemfolder + 'mountain.xml'
		simtime = 1000000
		resolu_factor = 1

		#update with additonal parameters. should have as many rows as parameters
		true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')

		likelihood_sediment = False
		#Set variables
		#m = 0.5
		m_min = 0.
		m_max = 2.
		#n = 1.
		n_min = 0.
		n_max = 2.
		#rain_real = 1.5
		rain_min = 0.
		rain_max = 3.
		#erod_real = 5.e-6
		erod_min = 3.e-6
		erod_max = 7.e-6
		#uplift_real = 50000
		uplift_min = 0.1 # X uplift_real
		uplift_max = 1.7 # X uplift_real
		
		minlim_param=[rain_min,erod_min,m_min,n_min,uplift_min]
		maxlim_param=[rain_max,erod_max,m_max,n_max,uplift_max]    
						
		## hence, for 4 regions of rain and 1 erod, plus other free parameters (p1, p2) [rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod, p1, p2 ]
		#if you want to freeze a parameter, keep max and min limits the same
		
		vec_param = np.random.uniform(minlim_param, maxlim_param) #  draw intial values for each of the free parameters

		stepsize_ratio  = 0.02 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

		stepratio_vec =  np.repeat(stepsize_ratio, vec_param.size) 
		stepratio_vec = [stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio, 0.02] 
		#stepratio_vec = [0.1, 0.02, 0.1, 0.1, 0.1]
		print("steps: ", stepratio_vec)
		num_param = vec_param.size
		erodep_coords=np.array([[5,5],[10,10],[20,20],[30,30],[40,40],[50,50],[25,25],[37,30],[44,27],[46,10]])
		#erodep_coords =  np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]]) # need to hand pick given your problem

		if (true_parameter_vec.shape[0] != vec_param.size ):
			print(' seems your true parameters file in data folder is not updated with true values of your parameters.  ')
			print('make sure that this is updated in case when you intro more parameters. should have as many rows as parameters ')
			return

	else:
		print('choose some problem  ')

	groundtruth_elev = np.loadtxt(problemfolder + 'data/final_elev.txt')
	groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
	groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')

	filename = ""
	run_nb = 0
	
	while os.path.exists(problemfolder +'results_%s' % (run_nb)):
		run_nb += 1
	if not os.path.exists(problemfolder +'results_%s' % (run_nb)):
		os.makedirs(problemfolder +'results_%s' % (run_nb))
		filename = (problemfolder +'results_%s' % (run_nb))
	run_nb_str = 'results_' + str(run_nb)



	num_successive_topo = 4
	sim_interval = np.arange(0, simtime+1, simtime/num_successive_topo) # for generating successive topography
	print(sim_interval)

	timer_start = time.time()

	#-------------------------------------------------------------------------------------
	#Create A a Patratellel temperatureing object instance 
	#-------------------------------------------------------------------------------------

	pt = ParallelTempering(vec_param, num_chains, maxtemp, samples, swap_interval,
	 filename, true_parameter_vec, num_param ,groundtruth_elev,  groundtruth_erodep_pts,
	 erodep_coords, simtime, sim_interval, resolu_factor, run_nb_str, xmlinput, surrogate_interval, surrogate_prob)


	directories = ['/posterior/pos_parameters','/posterior/pos_parameters/prop_list'
	,'/posterior/predicted_topo','/posterior/pos_likelihood','/posterior/surg_likelihood'
	,'/posterior/accept_list','/posterior/predicted_erodep','/pred_plots'
	, '/surrogate', '/surrogate/prediction_benchmark_data' ]
	
	for d in directories:
		pt.make_directory((filename)+ d)	
	#-------------------------------------------------------------------------------------
	# intialize the MCMC chains
	#-------------------------------------------------------------------------------------
	pt.initialize_chains(minlim_param, maxlim_param, stepratio_vec, likelihood_sediment, burn_in)

	#-------------------------------------------------------------------------------------
	#run the chains in a sequence in ascending order
	#-------------------------------------------------------------------------------------
	pos_param,likelihood_rep, accept_list, combined_erodep, pred_elev  = pt.run_chains()


	likelihood = likelihood_rep[:,0] # just plot proposed likelihood
	likelihood = np.asarray(np.split(likelihood, num_chains))

	# Plots
	plt.plot(likelihood.T)
	plt.savefig(filename+'/likelihood.png')
	plt.clf()
	plt.plot(accept_list.T)
	plt.savefig(filename+'/accept_list.png')
	plt.clf()

	for i in range(sim_interval.size):
		pos_ed  = combined_erodep[i, :, :] 
		erodep_mean = pos_ed.mean(axis=0)  
		erodep_std = pos_ed.std(axis=0)  
		pt.plot_erodep(erodep_mean, erodep_std, groundtruth_erodep_pts[i,:], sim_interval[i], filename) 

	pred_erodep = np.zeros((groundtruth_erodep_pts.shape[0], groundtruth_erodep_pts.shape[1] )) # just to get the right size

	for i in range(sim_interval.size): 
		pos_ed  = combined_erodep[i, :, :] # get final one for comparision
		pred_erodep[i,:] = pos_ed.mean(axis=0)   

	rmse, rmse_sed= pt.mean_sq_error(pred_erodep, pred_elev, groundtruth_elev, groundtruth_erodep_pts)

	pt.plot_boxplot(pos_param, filename)

	timer_end = time.time()

	with file(('%s/experiment_metrics.txt' % (filename)),'a') as outfile:
		outfile.write('\ntime elapsed (mins):,{0}'.format((timer_end-timer_start)/60))
		outfile.write('\nRMSE sediment:,{0}'.format(rmse_sed))
		outfile.write('\nRMSE overall:,{0}'.format(rmse))
	print('Sucessfully sampled') 
	print ('Time taken  in minutes = ', (timer_end-timer_start)/60)

if __name__ == "__main__": main()