'''
	This is a set of functions and classes to help the main prosses but they are pressindible.
'''
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progress.bar import  Bar
import random

import collections # reducer
import os # reducer
#import re # reducer
import pickle

def strToListF(cad:str, sep=' '):
	return [float(s) for s in cad.split(sep)]
def strToListI(cad:str, sep=' '):
	return [int(s) for s in cad.split(sep)]

def getSTime(time_sec):
	mon, sec = divmod(time_sec, 60)
	hr, mon = divmod(mon, 60)
	return "{:02.0f}:{:02.0f}:{:02.0f}".format(hr,mon,sec)

class MyBar(Bar):
	empty_fill = '.'
	fill = '='
	bar_prefix = ' ['
	bar_suffix = '] '
	value = 0
	hide_cursor = False
	width = 20
	suffix = '%(percent).1f%% - %(value).5f'
	
	def next(self, _v=None):
		if _v is not None:
			self.value = _v
		super(MyBar, self).next()

def colorizar(text):
	return '\033[91m' + text + '\033[0m'
def headerizar(text):
	return '\033[1m' + text + '\033[0m'

def getMyDict():
	return {'<emogy>':1, '<hashtag>':2, '<url>':3, '<risa>':4, '<signo>':5,
			'<ask>':6, '<phoria>':7, '<diag>':8, '<number>':9, '<date>':10,
			'<sent>':11, '<user>':12, '<frase>':13 }

def generate_dictionary_from_embedding(filename, dictionary, ret=True, logs=True, norm=False, message_logs=''):
	if logs:
		print ('# Loading:', colorizar(os.path.basename(filename)), message_logs)
	x = []
	band, l = False, 0

	mean, var, T = 0, 0, 0
	with open(filename, 'r', encoding='utf-8') as file:
		for ide, line in enumerate(file):
			li = line.split()

			if len(li) <= 2:
				print('#WARNING::', line, 'interpreted as head')
				continue
				
			if not band:
				x.append([0 for _ in range(len(li)-1)])
				my_d = getMyDict()
				l = len(my_d)
				for val in my_d:
					x.append([random.random() for _ in range(len(li)-1)])
					dictionary.update({val.lower(): my_d[val] })
				band = True

			a = [float(i) for i in li[1:]]
			x.append(a)
			
			mean += np.array(a, dtype=np.float32)
			var  += np.array(a, dtype=np.float32)**2
			T += 1

			dictionary.update({li[0].lower(): ide + l + 1})
	var  /= float(T)
	mean /= float(T)
	var -= mean ** 2
	var  = np.sqrt(var)
	mean = mean.reshape(1,mean.shape[0])
	var = var.reshape(1,var.shape[0])
	
	if ret:
		sol = np.array(x, np.float32)
		if norm:
			sol = (sol - mean) / var
		return sol

class TorchBoard(object):
	'''
		This is a naive board to plot the training and the evaluation phase
		using matplotlib
	'''
	def __init__(self):
		self.dict = {}
		self.labels = ['train', 'test', 'train_mse', 'test_mse', 'train_acc2', 'test_acc2', 'train_acc3', 'test_acc3']
		self.future_updt = True
		self.best_funct = None
		self.setFunct( max )
		self.best     = [None, None, None, None, None, None, None, None]
		self.best_p   = [0, 0, 0, 0, 0, 0, 0, 0]
		self.bestDot  = 'test'
	
	def setBestDotName(self, name):
		if name not in self.labels:
			print ('WARNING Ignore operation of \'setBestDotName\', name', name , 'not in ', self.labels)
			return 
		self.bestDot = name

	def setFunct(self, fun):
		'''
			Set the 'best' function to calculate the best point in the plot

			fun:
				a function pointer
		'''
		self.best_funct = fun

	def update(self, label, value, getBest=False):
		'''
		add a anotation
		label: str
			most be in ['train', 'test']
		value: number
		getBest: bool
			this make the funtion to return True if the value parameter given is the 'best' at the moment
			The 'best' is a function which calculate if the new value is in a sence better than the olders one, this is setted with the setFunct() method 
		'''
		if self.future_updt == False:
			return
		if label not in self.labels:
			print ('WARNING: the label {} its not in {}, the board will not be updated.'.format(
				label, self.labels))
			self.future_updt = False
			return
		pk = -1
		for i in range(len(self.labels)):
			if self.labels[i] == label:
				pk = i
				break

		if self.dict.get(label) == None:
			self.dict.update({label:[value]})
		else:
			self.dict[label].append(value)

		yo = False
		if self.best[pk] is None:
			yo = True
			self.best[pk] = value
		else:
			self.best[pk] = self.best_funct(self.best[pk], value)
			yo = self.best[pk] == value
		if yo:
			self.best_p[pk] = len(self.dict[label]) - 1
		if getBest:
			return yo

	def show(self, saveroute, plot_smood=False, pk_save=False, live=False):
		'''
			Save the plot

			saverroute: str
				path to save the plot
			plt_smood: bool
				If is True, plot a dotted curve repressentig the smood real curve.
		'''
		fig , axes = plt.subplots()
		for i,l in enumerate(self.dict):
			if self.best[i] is None:
				continue
			
			y = self.dict[l]
			if len(y) <= 1:
				continue
			lab = str(self.best[i])
			if len(lab) > 7:
				lab = lab[:7]
			axes.plot(range(len(y)), y, label=l + ' ' + lab)
			if l == self.bestDot:
				axes.scatter([self.best_p[i]], [self.best[i]])
			if plot_smood:
				w = 3
				y_hat = [ np.array(y[max(i-w,0):min(len(y),i+w)]).mean() for i in range(len(y))]
				axes.plot(range(len(y)), y_hat, ls='--', color='gray')

		fig.legend()
		fig.savefig(saveroute)

		if live:
			plt.show()

		del axes
		del fig

		if pk_save:
			import pickle
			file = open(saveroute+'.pk', 'wb')
			pickle.dump(self, file)
			file.close()
	
	def pkLoad(self, loadroute):
		import pickle
		file = open(loadroute, 'rb')
		temporal = pickle.load(file)
		file.close()

		self.dict = temporal.dict
		self.future_updt = temporal.future_updt
		self.best_funct = temporal.best_funct
		self.best     = temporal.best
		self.best_p   = temporal.best_p
		self.bestDot  = temporal.bestDot

		del temporal

def makeParameters(params, file):
	with open(file, 'w') as F:
		for name in params:
			F.write(name + " : " + str(params[name]) + '\n')

def parceParameter(file):
	__sol = []
	with open(file, 'r') as F:
		for line in F.readlines():
			line = line.replace(' ', '').replace('\n', '').split(':') 
			name, value = str(line[0]), line[1]
			__sol.append((name, value))
	return dict(__sol)

def plot_loos_np(path, smood=False, win=10):
	losses_ = np.load(path)
	plt.figure(figsize=(8,6))

	if smood:
		mean_losses_ = np.zeros_like(losses_)
		for i in range(losses_.shape[0]):
			i_min, i_max = max(i-win, 0), min(losses_.shape[0], i+win)
			mean_losses_[i,0] = losses_[i_min:i_max,0].mean()
			mean_losses_[i,1] = losses_[i_min:i_max,1].mean()
			mean_losses_[i,2] = losses_[i_min:i_max,2].mean()
		plt.plot(np.log(losses_[:,0]),label='Q loss', alpha=0.3, c='b')
		plt.plot(np.log(losses_[:,1]),label='Forward loss', alpha=0.3, c='orange')
		plt.plot(np.log(losses_[:,2]),label='Inverse loss', alpha=0.3, c='green')
		plt.plot(np.log(mean_losses_[:,0]), c='b')
		plt.plot(np.log(mean_losses_[:,1]), c='orange')
		plt.plot(np.log(mean_losses_[:,2]), c='green')
	else:
		plt.plot(np.log(losses_[:,0]),label='Q loss')
		plt.plot(np.log(losses_[:,1]),label='Forward loss')
		plt.plot(np.log(losses_[:,2]),label='Inverse loss')
	plt.legend()
	plt.show()

def makeVocabFromData(filepath):
	'''
	Make a vocabulary from a file

	input:
		filepath: str
			This file is splited with a space separator (' ') and the words are returned

	output:
		vocabulary: dict
	'''
	c = None
	with open(filepath, 'r', encoding='utf-8') as f:
		line = f.read().replace('\n', ' ')
		c = collections.Counter(line.split())

	return dict([(i, 5) for i in sorted(c, reverse=True)])

def projectData2D(data_path:str, save_name='2Data', drops = ['is_humor','humor_rating', 'id'], use_centers=False):
	'''
		Project the vetors in 2d plot

		data_path:str most be a cvs file
	'''
	from sklearn.manifold import TSNE 
	from sklearn.decomposition import PCA
	
	data = pd.read_csv(data_path)

	np_data = data.drop(drops, axis=1).to_numpy().tolist()
	np_data = [i for i in map(lambda x: [float(v) for v in x[0].split()], np_data)]
	np_data = np.array(np_data, dtype=np.float32) 

	L1, L2 = 0, 0
	if use_centers:
		# L1, L2 = [], []
		# P = [['neg_center.txt', L1], ['pos_center.txt', L2]]
		# for l,st in P:
		# 	with open(os.path.join('data', l), 'r') as file:
		# 		lines = file.readlines()
		# 		lines = np.array([[float(v) for v in x.split()] for x in lines], dtype=np.float32)
		# 		st.append(lines)
		# L1 = np.concatenate(L1, axis=0)
		# L2 = np.concatenate(L2, axis=0)	
		L1 = np.load(os.path.join('data', 'neg_center.npy'))
		L2 = np.load(os.path.join('data', 'pos_center.npy'))

		np_data = np.concatenate([np_data, L1, L2], axis=0)
		L1, L2 = L1.shape[0], L2.shape[0]
	print ('# Projecting', colorizar(os.path.basename(data_path)), 'in 2d vectors', end='')
	X_embb = TSNE(n_components=2).fit_transform(np_data)
	# X_embb = PCA(n_components=2, svd_solver='full').fit_transform(np_data)
	#X_embb = TruncatedSVD(n_components=2).fit_transform(np_data)
	print ('  Done!')
	del np_data
	
	D_1, D_2 = [], []
	fig , axes = plt.subplots()

	for i in range(len(data)):
		if int(data.loc[i, 'is_humor']) == 0:
			D_1.append([X_embb[i,0], X_embb[i,1]])
		else:
			D_2.append([X_embb[i,0], X_embb[i,1]])
	D_1, D_2 = np.array(D_1), np.array(D_2)
	axes.scatter(D_1[:,0], D_1[:,1], label=r'$N~class$', color=(255/255, 179/255, 128/255, 1.0))
	axes.scatter(D_2[:,0], D_2[:,1], label=r'$P~class$', color=(135/255, 222/255, 205/255, 1.0))
	
	if L1 > 0:
		X_embb_1 = X_embb[-(L1+L2):-L2]
		X_embb_2 = X_embb[-L2:]

		axes.scatter(X_embb_1[:,0], X_embb_1[:,1], label=r'$N_{Set}$', color=(211/255, 95/255, 95/255, 1.0), marker='s')
		axes.scatter(X_embb_2[:,0], X_embb_2[:,1], label=r'$P_{Set}$', color=(0/255, 102/255, 128/255, 1.0), marker='s')
		
		del X_embb_1
		del X_embb_2
	del X_embb

	fig.legend()
	fig.tight_layout()
	axes.axis('off')
	fig.savefig(os.path.join('out', save_name+'.png'))
	print ('# Image saved in', colorizar(os.path.join('out', save_name+'.png')))
	# plt.show()
	
	del fig
	del axes
	
	
