### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time
from itertools import product

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model modules
from parameters import *
import stimulus
import AdamOpt
from model import Model, get_perf

# Match GPU IDs to nvidia-smi command
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Load saved data
data = pickle.load(open('./savedir/LSTM_SL_Vanilla_pLIN.pkl', 'rb'))

data['par'].update({'batch_size':1024})
update_parameters(data['par'], verbose=False)

weight_dict = {k+'_init' : v for k, v in data['weights'].items()}
update_parameters(weight_dict, verbose=False, update_deps=False)
print('Saved data loaded.\n')

# Isolate requested GPU
try:
	gpu_id = sys.argv[1]
except:
	gpu_id = None

if gpu_id is not None:
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# Reset Tensorflow graph before running anything
tf.reset_default_graph()

# Define all placeholders
plc_x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_input']], 'stim')
plc_y = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_output']], 'out')
plc_m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')
plc_g = tf.placeholder(tf.float32, [par['n_hidden']], 'gating')
plc_d = tf.placeholder_with_default(0.5, [], 'lin_dropout')
plc_l = tf.placeholder_with_default(np.ones([par['batch_size'],par['n_linear']]).astype(np.float32), [par['batch_size'],par['n_linear']], 'lesion')

# Set up stimulus and accuracy recording
stim = stimulus.MultiStimulus()

# Start Tensorflow session
with tf.Session() as sess:

	# Select CPU or GPU
	device = '/cpu:0' if gpu_id is None else '/gpu:0'
	with tf.device(device):
		model = Model(plc_x, plc_y, plc_m, plc_g, plc_d, plc_l)

	# Initialize variables and start the timer
	sess.run(tf.global_variables_initializer())

	for task in range(par['n_tasks']):

		print('\nAnalyzing task {}...'.format(task))

		name, stim_in, y_hat, mk, _ = stim.generate_trial(task)
		feed_dict = {plc_x:stim_in, plc_y:y_hat, plc_g:par['gating'][task], plc_m:mk, plc_d:1.}
		loss, base_rec_loss, output, lin, recon = \
			sess.run([model.pol_loss, model.rec_loss, model.output, \
			model.lin, model.recon], feed_dict=feed_dict)

		acc = get_perf(y_hat, output, mk)
		output = np.stack(output, axis=0)
		linear = np.stack(lin, axis=0)		# time x trials x neurons

		print('Task {} | Loss: {:5.3f} | Acc: {:5.3f} |'.format(task, loss, acc))

		# Look at patterns of activity during certain types of trials
		print('Rendering directional activity patterns.')
		fig, ax = plt.subplots(2,4,figsize=(8,4))
		fields = []
		for d in range(par['num_motion_dirs']):
			dir_inds = np.where(y_hat[-1,:,d] == 1.)[0]
			neural_field = np.mean(linear[:,dir_inds,:], axis=1)
			
			idx = d%2
			idy = d//2
			fields.append(neural_field.T)
			ax[idx,idy].imshow(neural_field.T, aspect='auto')
			ax[idx,idy].set_title('Direction {}'.format(d))
			ax[idx,idy].set_xticks([])
			ax[idx,idy].set_yticks([])

		ax[0,0].set_ylabel('Neurons')
		ax[1,0].set_ylabel('Neurons')
		ax[1,0].set_xlabel('Time')

		fig.suptitle('Task {} : Directional Activity Patterns'.format(task))
		plt.savefig('./analysis/task{}_all_neural_activity.png'.format(task), bbox_inches='tight')
		plt.clf()
		plt.close()

		fig, ax = plt.subplots(2,4,figsize=(8,4))
		field_mean = np.mean(fields, axis=0)
		for d in range(par['num_motion_dirs']):
			idx = d%2
			idy = d//2

			ax[idx,idy].imshow(field_mean - fields[d], aspect='auto')
			ax[idx,idy].set_title('Direction {}'.format(d))
			ax[idx,idy].set_xticks([])
			ax[idx,idy].set_yticks([])

		ax[0,0].set_ylabel('Neurons')
		ax[1,0].set_ylabel('Neurons')
		ax[1,0].set_xlabel('Time')

		fig.suptitle('Task {} : Difference from Mean of Directional Activity Patterns'.format(task))
		plt.savefig('./analysis/task{}_all_neural_activity_var_from_mean.png'.format(task), bbox_inches='tight')
		plt.clf()
		plt.close()


		# Look at average activity for EACH NEURON at EACH TIME STEP
		# for each of the 8 motion directions

		### One plot for each neuron
		### Essentially the same as the previous plots, but now separated by neuron
		###   and with all directions in the same figure
		### Do eight plots of 4 x 4 subplots for 128 neurons in total
		print('Processing linear neuron tunings.')
		num_plots = np.ceil(par['n_linear']/16).astype(np.int32)
		dir_inds = [np.where(y_hat[-1,:,d] == 1.)[0] for d in range(par['num_motion_dirs'])]
		for p in range(num_plots):
			neurons = np.arange(p*16,(p+1)*16)

			fig, ax = plt.subplots(4,4,figsize=(8,8))
			for n in neurons:
				idx = (n%16)//4
				idy = (n%16)%4

				# linear = [time x trials x neurons]
				responses = []
				for d in range(par['num_motion_dirs']):
					neuron_response = np.mean(linear[:,dir_inds[d],n], axis=1)
					responses.append(neuron_response)

				responses = np.stack(responses, axis=0)
				
				ax[idx,idy].set_title('Neuron {}'.format(n))
				ax[idx,idy].set_xticks([])
				ax[idx,idy].set_yticks([])
				ax[idx,idy].imshow(responses, aspect='auto')

			for i in range(4):
				ax[i,0].set_ylabel('Directions')
			ax[3,0].set_xlabel('Time')

			fig.suptitle('Task {} : Neuron Directional Tuning'.format(task))
			plt.savefig('./analysis/task{}_neuron{}-{}_tuning.png'.format(task, neurons.min(), neurons.max()), bbox_inches='tight')
			plt.clf()
			plt.close()

		print('Running linear neuron double lesioning against reconstruction.')
		lesioned_losses = np.zeros([par['n_linear'], par['n_linear']])
		for n in range(par['n_linear']):
			for m in range(n,par['n_linear']):
				print(n, m)

				lmask = np.ones([par['batch_size'],par['n_linear']]).astype(np.float32)
				lmask[:,n] = 0.
				lmask[:,m] = 0.

				feed_dict = {plc_x:stim_in, plc_g:par['gating'][task], plc_m:mk, plc_d:1., plc_l:lmask}
				recon_loss, = sess.run([model.rec_loss], feed_dict=feed_dict)
				lesioned_losses[n,m] = recon_loss/base_rec_loss
				lesioned_losses[m,n] = recon_loss/base_rec_loss

		tl = lambda i : str(i) if i in np.arange(0,par['n_linear'],8) else ''
		tick_labels = [tl(n) for n in range(par['n_linear'])]

		fig, ax = plt.subplots(1,1,figsize=(8,8))
		ax.imshow(lesioned_losses, aspect='auto', clim=(0, lesioned_losses.max()))
		ax.set_xlabel('Lesioned Neuron 1')
		ax.set_ylabel('Lesioned Neuron 2')
		ax.set_xticks(np.arange(0,par['n_linear'],8))
		ax.set_yticks(np.arange(0,par['n_linear'],8))
		ax.set_title('Task {} : Percentage Reconstruction Loss After Lesioning Linear Units'.format(task))
		plt.savefig('./analysis/task{}_linear_double-lesion.png'.format(task), bbox_inches='tight')
		plt.clf()
		plt.close()



	print('Analysis complete!')