from __future__ import division
import os, helper, time, scipy.io #,sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

checkpoint_name = "result_256p_reza2"

# Activation Function
def lrelu(x):
	return tf.maximum(0.2 * x, x)

# Helper function for creating NN layers
def build_net(ntype, nin, nwb=None, name=None):
	if ntype == 'conv':
		return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
	elif ntype == 'pool':
		return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Gets weights and biases from the raw VGG19 network
def get_weight_bias(vgg_layers, i):
	weights = vgg_layers[i][0][0][2][0][0]
	weights = tf.constant(weights)
	bias = vgg_layers[i][0][0][2][0][1]
	bias = tf.constant(np.reshape(bias, (bias.size)))
	return weights, bias

# Build the VGG19 Network for this problem
def build_vgg19(input, reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()
	net = {}
	vgg_rawnet = scipy.io.loadmat('VGG_Model/imagenet-vgg-verydeep-19.mat')
	vgg_layers = vgg_rawnet['layers'][0]
	net['input'] = input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
	net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
	net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
	net['pool1'] = build_net('pool', net['conv1_2'])
	net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
	net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
	net['pool2'] = build_net('pool', net['conv2_2'])
	net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
	net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
	net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
	net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
	net['pool3'] = build_net('pool', net['conv3_4'])
	net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
	net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
	net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
	net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
	net['pool4'] = build_net('pool', net['conv4_4'])
	net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
	net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
	net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32), name='vgg_conv5_3')
	net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34), name='vgg_conv5_4')
	net['pool5'] = build_net('pool', net['conv5_4'])
	return net

# Generator with down sampling to the proper size for each layer.
def recursive_generator(label, sp):
	dim = 512 if sp >= 128 else 1024
	if sp == 4:
		input = label
	else:
		downsampled = tf.image.resize_area(label, (sp//2, sp), align_corners=False)
		input = tf.concat([tf.image.resize_bilinear(recursive_generator(downsampled, sp // 2), (sp, sp * 2), align_corners=True), label], 3)
	net = slim.conv2d(input, dim, [3, 3], rate=1, normalizer_fn=slim.layer_norm, activation_fn=lrelu, scope='g_' + str(sp) + '_conv1')
	net = slim.conv2d(net, dim, [3, 3], rate=1, normalizer_fn=slim.layer_norm, activation_fn=lrelu, scope='g_' + str(sp) + '_conv2')
	if sp==256:
		net = slim.conv2d(net, 27, [1, 1], rate = 1, activation_fn=None, scope='g_' + str(sp) + '_conv100')
		net = (net + 1.0) / 2.0 * 255.0
		split0, split1, split2 = tf.split(tf.transpose(net, perm=[3, 1, 2, 0]), num_or_size_splits=3, axis=0)
		net=tf.concat([split0, split1, split2], 3)
	return net

# Loss function for a singe layer in VGG19
def compute_error(real, fake, label):
	return tf.reduce_mean(label * tf.expand_dims(tf.reduce_mean(tf.abs(fake - real), reduction_indices=[3]), -1), reduction_indices=[1, 2]) # diversity loss

# Create Labels tensor with appropriate dimensions for the network
def create_labels(label):
	return np.concatenate((label, np.expand_dims(1 - np.sum(label, axis=3), axis=3)), axis=3)

# Create a SciPy Image from the output of the network
def create_scipy_img(output):
	bounded_output = np.minimum(np.maximum(output, 0.0), 255.0)
	upper = np.concatenate((bounded_output[0, :, :, :], bounded_output[1, :, :, :], bounded_output[2, :, :, :]), axis=1)
	middle = np.concatenate((bounded_output[3, :, :, :], bounded_output[4, :, :, :], bounded_output[5, :, :, :]), axis=1)
	bottom = np.concatenate((bounded_output[6, :, :, :], bounded_output[7, :, :, :], bounded_output[8, :, :, :]), axis=1)
	return scipy.misc.toimage(np.concatenate((upper, middle, bottom), axis=0), cmin=0, cmax=255)

def print_message(msg):
	if (USE_SYS_STDOUT):
		sys.stdout.write('%s\n' % msg)
	else:
		print(msg)

print_message("PID = %d" % os.getpid())
sess = tf.Session(config = config)
sp = 256 # spatial resolution: 256x512
with tf.variable_scope(tf.get_variable_scope()):
	label = tf.placeholder(tf.float32, [None, None, None, 20])
	real_image = tf.placeholder(tf.float32, [None, None, None, 3])
	fake_image = tf.placeholder(tf.float32, [None, None, None, 3])
	generator = recursive_generator(label, sp)
	weight = tf.placeholder(tf.float32)
	vgg_real = build_vgg19(real_image)
	vgg_fake = build_vgg19(generator, reuse=True)
	p0 = compute_error(vgg_real['input'], vgg_fake['input'], label)
	p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2'], label) / 1.6
	p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2'], tf.image.resize_area(label, (sp // 2, sp))) / 2.3
	p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2'], tf.image.resize_area(label, (sp // 4, sp // 2))) / 1.8
	p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2'], tf.image.resize_area(label, (sp // 8, sp // 4))) / 2.8
	p5 = compute_error(vgg_real['conv5_2'], vgg_fake['conv5_2'], tf.image.resize_area(label, (sp // 16, sp // 8))) * 10 / 0.8 # weights lambda are collected at 100th epoch
	content_loss = p0 + p1 + p2 + p3 + p4 + p5
	# content_loss = p5
	G_loss = tf.reduce_sum(tf.reduce_min(content_loss, reduction_indices=0)) * 0.999 + tf.reduce_sum(tf.reduce_mean(content_loss, reduction_indices=0)) * 0.001
lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
saver = tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_name)
# Load pre-trained model
if ckpt:
	print_message('loaded ' + ckpt.model_checkpoint_path)
	saver.restore(sess, ckpt.model_checkpoint_path)


# https://docs.python.org/3/library/tkinter.html
import os
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

cwd = os.path.dirname(__file__)

class Application(tk.Frame):
	def __init__(self, master=None):
		super().__init__(master)
		self.master = master
		self.layoutPath = os.path.join(cwd, 'red.png')
		self.outputPath = os.path.join(cwd, 'output')
		self.outputFile = os.path.join(cwd, 'blue.png')
		self.imgH = 256
		self.imgW = 512
		self.pack()
		self.createInputSelector()
		self.createOutputGeneratorBtn()
		self.createLayoutLabel()
		self.createOutputLabel()
		# Running against test data
		if not os.path.isdir(self.outputPath):
			os.makedirs(self.outputPath)


	def createInputSelector(self):
		self.browseBtn = tk.Button(self)
		self.browseBtn["text"] = "Select Layout"
		self.browseBtn["command"] = self.selectLayout
		self.browseBtn.pack(side="left")

	def createOutputGeneratorBtn(self):
		self.browseBtn = tk.Button(self)
		self.browseBtn["text"] = "RUN!"
		self.browseBtn["command"] = self.generateOutput
		self.browseBtn.pack(side="right")

	def createLayoutLabel(self):
		self.layoutImg = ImageTk.PhotoImage(Image.open(self.layoutPath))
		self.layoutLabel = tk.Label(window, image = self.layoutImg, height = self.imgH, width = self.imgW)
		self.layoutLabel.pack(side="left")

	def createOutputLabel(self):
		self.outoutImg = ImageTk.PhotoImage(Image.open(self.outputFile))
		self.outputLabel = tk.Label(window, image = self.outoutImg, height=self.imgH, width=self.imgW)
		self.outputLabel.pack(side="right")

	def selectLayout(self):
		self.layoutPath = filedialog.askopenfilename(initialdir = os.path.join(cwd, 'input'), title = "Select Layout", filetypes = (("PNG Files", "*.png"),))
		if (not self.layoutPath):
			return
		self.layoutPath = os.path.abspath(self.layoutPath)
		self.updateLayoutWithSelection()

	def updateLayoutWithSelection(self):
		self.layoutImg = ImageTk.PhotoImage(Image.open(self.layoutPath))
		self.layoutLabel.configure(image = self.layoutImg)
		self.layoutLabel.image = self.layoutImg

	def generateOutput(self):
		self.outputFile = os.path.join(self.outputPath, os.path.basename(self.layoutPath))
		self.updateOutputWithResult()

	def updateOutputWithResult(self):
		semantic = helper.get_semantic_map(self.layoutPath)
		output = sess.run(generator, feed_dict={label:create_labels(semantic)})
		create_scipy_img(output).save(self.outputFile)
		self.outoutImg = ImageTk.PhotoImage(Image.open(self.outputFile))
		self.outputLabel.configure(image = self.outoutImg)
		self.outputLabel.image = self.outoutImg


window = tk.Tk()
window.title("CS636 Project - Reza Teshnizi - UIN: 122006181")
window.geometry("1100x400")
app = Application(master=window)
app.mainloop()
