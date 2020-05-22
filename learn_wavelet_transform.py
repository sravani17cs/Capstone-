import tensorflow as tf
import numpy as np
import sys
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from tensorflow import keras
from find_peak import read_data, digitalToRaw
from wave_denoising import denoise
from wave_denoising import transform

leadLabel = {
	'MLII':0,
	'V1':1,
	'V2':2,
	'V4':3,
	'V5':4,
}

#label = {
#	"N":0,
#	"A":1,
#}
label = {
	"N":0,
	"L":1,
	"R":2,
	"a":3,
	"V":4,
	"F":5,
	"J":6,
	"A":7,
	"S":8,
	"E":9,
	"j":10,
	"/":11,
	"Q":12,
	"~":13,
	"|":14,
	"s":15,
	"T":16,
	"*":17,
	"D":18,
	'"':19,
	"=":20,
	"p":21,
	"B":22,
	"^":23,
	"t":24,
	"+":25,
	"u":26,
	"?":27,
	"!":28,
	"[":29,
	"]":30,
	"e":31,
	"n":32,
	"@":33,
	"x":34,
	"f":35,
	"(":36,
	")":37,
	"r":38
}

# label = {
	# "N":"Normal beat",
	# "L":"Left bundle branch block beat",
	# "R":"Right bundle branch block beat",
	# "a":"Aberrated atrial premature beat",
	# "V":"Premature ventricular contraction",
	# "F":"Fusion of ventricular and normal beat",
	# "J":"Nodal (junctional) premature beat",
	# "A":"Atrial premature contraction",
	# "S":"Premature or ectopic supraventricular beat",
	# "E":"Ventricular escape beat",
	# "j":"Nodal (junctional) escape beat",
	# "/":"Paced beat",
	# "Q":"Unclassifiable beat",
	# "~":"Signal quality change",
	# "|":"Isolated QRS-like artifact",
	# "s":"ST change",
	# "T":"T-wave change",
	# "*":"Systole",
	# "D":"Diastole",
	# '"':"Comment annotation",
	# "=":"Measurement annotation",
	# "p":"P-wave peak",
	# "B":"Left or right bundle branch block",
	# "^":"Non-conducted pacer spike",
	# "t":"T-wave peak",
	# "+":"Rhythm change",
	# "u":"U-wave peak",
	# "?":"Learning",
	# "!":"Ventricular flutter wave",
	# "[":"Start of ventricular flutter/fibrillation",
	# "]":"End of ventricular flutter/fibrillation",
	# "e":"Atrial escape beat",
	# "n":"Supraventricular escape beat",
	# "@":"Link to external data (aux_note contains URL)",
	# "x":"Non-conducted P-wave (blocked APB)",
	# "f":"Fusion of paced and normal beat",
	# "(":"Waveform onset",
	# ")":"Waveform end",
	# "r":"R-on-T premature ventricular contraction"
# }

class TimeHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.times = []
		self.batch_count = 0
		
	def on_batch_begin(self,batch,logs):
		# print(batch,logs)
		pass
		
	def on_batch_end(self,batch,logs):
		# print(batch,logs)
		pass

	def on_epoch_begin(self, epoch, logs):
		self.epoch_time_start = time.time()

	def on_epoch_end(self, epoch, logs):
		t = time.time() - self.epoch_time_start
		print(epoch, logs, t)
		self.times.append(t)
callback = TimeHistory()
# load data
filter = False
filter_type = "MLII"
epochs = 10
batch_size = 512
if True:
	print("Reading data...")
	with open("data_90_pickle", 'rb') as f:
		leadType, data, gain, base, symbol = pickle.load(f)
	# data = np.array([i-(i[0]-1024) for i in data]) # center all points to start at 1024
	
	# doing "feature extraction"
	# print('Wavelet denosing with 2 levels, bior2.4')
	# data = np.array([denoise(i) for i in data])
	
	# data = normalize(data)
	
	# data = np.array([digitalToRaw(data[i],gain[i],base[i]) for i in range(len(data))])
	# data = data.reshape((-1,inputSize[0],inputSize[1]))
	if len(label) == 2:
		symbol = np.array([0 if i == 'N' else 1 for i in symbol])
		# symbol = keras.utils.to_categorical(symbol)
	else:
		symbol = [label[i] for i in symbol]
	# filter wanted type
	if filter:
		if filter_type in leadLabel:
			index = [i for i in range(len(leadType)) if leadType[i] == filter_type]
			print(f"Data has been filtered with only {filter_type} data")
			data = np.array([data[i] for i in index])
			symbol = np.array([symbol[i] for i in index])
			leadType = np.array([leadType[i] for i in index])
			# data = data.reshape((-1,inputSize[0],inputSize[1]))
		else:
			print(f"Filter type {filter_type} is invalid")
			exit(1)
	# symbol = keras.utils.to_categorical(symbol,num_classes=len(label))
	data = np.array([[signal]+transform(signal) for signal in data])

def residual():
	# paper small
	# target validation accuracy with 2 label, MLII only: high 90s
	input_sizes = (90, 49, 29, 19, 14, 14)
	input_list = []
	output_list = []
	for i in range(5):
		input = keras.layers.Input(input_sizes[i])
		input_list.append(input)
		x = keras.layers.Reshape((input_sizes[i], 1))(input)
		
		# shortcut = keras.layers.AveragePooling1D()(x)
		x = keras.layers.Conv1D(4, (10),padding='same')(x)
		# shortcut = keras.layers.AveragePooling1D()(x)
		# x = keras.layers.Conv1D(16, (5),padding='same')(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.Activation('relu')(x)
		x = keras.layers.MaxPooling1D()(x)
		# x = keras.layers.Dropout(0.5)(x)
		x = keras.layers.Flatten()(x)
		output_list.append(x)
	
	merge = keras.layers.concatenate(output_list)#, axis=1)
	x = keras.layers.Dense(256,activation='relu')(merge)
	x = keras.layers.Dense(128,activation='relu')(x)
	x = keras.layers.Dense(64,activation='relu')(x)
	if len(label) == 2:
		x = keras.layers.Dense(1, activation='sigmoid')(x)
	else:
		x = keras.layers.Dense(len(label),activation='softmax')(x)


	model = keras.Model(inputs=input_list,outputs=x)

	# model.summary()
	return model
	
stats = []
kfold = KFold(n_splits=5, shuffle=True)
i = 0
print(f'Label size: {len(label)}')
for train_index, test_index in kfold.split(data):
	print(f'Trial {i+1}')
	size = np.arange(len(data))

	data_train, data_test = data[train_index], data[test_index]
	symbol_train = np.array([symbol[i] for i in train_index])
	symbol_test = np.array([symbol[i] for i in test_index])
	
	model = residual()
	# model.summary()
	model.compile(optimizer='adadelta',
		loss='binary_crossentropy' if len(label) == 2 else 'sparse_categorical_crossentropy',
		# loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])
	# with tf.device("/CPU:0"):
	model.fit([[i[j] for i in data_train] for j in range(5)],symbol_train,validation_split=0.1, batch_size=batch_size, epochs=epochs, verbose=2)
	loss, acc = model.evaluate([[i[j] for i in data_test] for j in range(5)],symbol_test, batch_size=32, verbose=2)
		
	print(f'Loss: {loss:<2.2f}', f'Accuracy {acc*100:<2.2f}')
	stats.append((loss,acc))
	del model
	i += 1
	
avg_loss, avg_acc = np.mean(stats,axis=0)
print("Label Size:", len(label))
for i in range(len(stats)):
	loss, acc = stats[i]
	print(f"Trial {i}: Loss: {loss:0.3f} Acc: {acc:0.3f}")
print(f'Avg. Loss: {avg_loss:<2.2f}', f'Avg. Accuracy {avg_acc*100:<2.2f}')
