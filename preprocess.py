# from os import listdir
# from numpy import asarray
# from numpy import save
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# # define location of dataset
# folder = 'train/'
# photos, labels = list(), list()
# # enumerate files in the directory
# for file in listdir(folder):
# 	# determine class
# 	output = 0.0
# 	if file.startswith('parrot'):
# 		output = 1.0
# 	# load image
# 	photo = load_img(folder + file, target_size=(200, 200))
# 	# convert to numpy array
# 	photo = img_to_array(photo)
# 	# store
# 	photos.append(photo)
# 	labels.append(output)
# # convert to a numpy arrays
# photos = asarray(photos)
# labels = asarray(labels)
# print(photos.shape, labels.shape)
# # save the reshaped photos
# save('ant_vs_parrot_photos.npy', photos)
# save('ant_vs_parrot_labels.npy', labels)

# from numpy import load
# photos = load('ant_vs_parrot_photos.npy')
# labels = load('ant_vs_parrot_labels.npy')
# print(photos.shape, labels.shape)

# from os import makedirs
dataset_home = 'dataset_ant_vs_parrot_final/'
# subdirs = ['train/', 'test/']
# for subdir in subdirs:
# 	# create label subdirectories
# 	labeldirs = ['ant/', 'parrot/']
# 	for labldir in labeldirs:
# 		newdir = dataset_home + subdir + labldir
# 		makedirs(newdir, exist_ok=True)

# from random import seed
# from random import random
# from os import listdir
# from shutil import copyfile
# seed(1)
# # define ratio of pictures to use for validation
# val_ratio = 0.25
# # copy training dataset images into subdirectories
# src_directory = 'train/'
# for file in listdir(src_directory):
# 	src = src_directory + '/' + file
# 	dst_dir = 'train/'
# 	if random() < val_ratio:
# 		dst_dir = 'test/'
# 	if file.startswith('ant'):
# 		dst = dataset_home + dst_dir + 'ant/'  + file
# 		copyfile(src, dst)
# 	elif file.startswith('parrot'):
# 		dst = dataset_home + dst_dir + 'parrot/'  + file
# 		copyfile(src, dst)


################################################################ VGG Model ##########################################################################
# vgg16 model used for transfer learning on the dogs and cats dataset
# import sys
# from matplotlib import pyplot
# from keras.utils import to_categorical
# from keras.applications.vgg16 import VGG16
# from keras.models import Model
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.optimizers import SGD
# from keras.preprocessing.image import ImageDataGenerator
 
# # define cnn model
# def define_model():
# 	# load model
# 	model = VGG16(include_top=False, input_shape=(224, 224, 3))
# 	# mark loaded layers as not trainable
# 	for layer in model.layers:
# 		layer.trainable = False
# 	# add new classifier layers
# 	flat1 = Flatten()(model.layers[-1].output)
# 	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
# 	output = Dense(1, activation='sigmoid')(class1)
# 	# define new model
# 	model = Model(inputs=model.inputs, outputs=output)
# 	# compile model
# 	opt = SGD(lr=0.001, momentum=0.9)
# 	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
# 	return model
 
# # plot diagnostic learning curves
# def summarize_diagnostics(history):
# 	# plot loss
# 	pyplot.subplot(211)
# 	pyplot.title('Cross Entropy Loss')
# 	pyplot.plot(history.history['loss'], color='blue', label='train')
# 	pyplot.plot(history.history['val_loss'], color='orange', label='test')
# 	# plot accuracy
# 	pyplot.subplot(212)
# 	pyplot.title('Classification Accuracy')
# 	pyplot.plot(history.history['accuracy'], color='blue', label='train')
# 	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
# 	# save plot to file
# 	filename = sys.argv[0].split('/')[-1]
# 	pyplot.savefig(filename + '_plot.png')
# 	pyplot.close()
 
# # run the test harness for evaluating a model
# def run_test_harness():
# 	# define model
# 	model = define_model()
# 	# create data generator
# 	datagen = ImageDataGenerator(featurewise_center=True)
# 	# specify imagenet mean values for centering
# 	datagen.mean = [123.68, 116.779, 103.939]
# 	# prepare iterator
# 	train_it = datagen.flow_from_directory('dataset_ant_vs_parrot_final/train/',
# 		class_mode='binary', batch_size=64, target_size=(224, 224))
# 	test_it = datagen.flow_from_directory('dataset_ant_vs_parrot_final/test/',
# 		class_mode='binary', batch_size=64, target_size=(224, 224))
# 	# fit model
# 	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
# 		validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)
# 	# evaluate model
# 	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
# 	print('> %.3f' % (acc * 100.0))
# 	# learning curves
# 	summarize_diagnostics(history)

 
# # entry point, run the test harness
# run_test_harness()

#################################################################################################################################################################

##### VGG results ##################

# 2021-04-23 16:47:42.419855: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
# 2021-04-23 16:47:42.419896: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
# 2021-04-23 16:47:44.319060: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
# 2021-04-23 16:47:44.329094: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
# 2021-04-23 16:47:44.329181: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
# 2021-04-23 16:47:44.329251: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (prasad): /proc/driver/nvidia/version does not exist
# 2021-04-23 16:47:44.330083: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2021-04-23 16:47:44.332183: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
# Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
# 58892288/58889256 [==============================] - 93s 2us/step
# Found 58 images belonging to 2 classes.
# Found 22 images belonging to 2 classes.
# /home/prasad/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:1844: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
#   warnings.warn('`Model.fit_generator` is deprecated and '
# 2021-04-23 16:49:20.610835: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
# 2021-04-23 16:49:20.660663: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 1800000000 Hz
# Epoch 1/10
# 2021-04-23 16:49:22.157898: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 745013248 exceeds 10% of free system memory.
# 2021-04-23 16:49:22.665515: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 745013248 exceeds 10% of free system memory.
# 2021-04-23 16:49:28.570423: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 186253312 exceeds 10% of free system memory.
# 2021-04-23 16:49:28.956021: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 372506624 exceeds 10% of free system memory.
# 2021-04-23 16:49:29.421089: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 372506624 exceeds 10% of free system memory.
# 1/1 [==============================] - 22s 22s/step - loss: 4.8013 - accuracy: 0.6034 - val_loss: 8.9808 - val_accuracy: 0.5455
# Epoch 2/10
# 1/1 [==============================] - 14s 14s/step - loss: 8.4948 - accuracy: 0.5862 - val_loss: 7.9074e-12 - val_accuracy: 1.0000
# Epoch 3/10
# 1/1 [==============================] - 17s 17s/step - loss: 1.2689e-05 - accuracy: 1.0000 - val_loss: 0.1978 - val_accuracy: 0.9545
# Epoch 4/10
# 1/1 [==============================] - 18s 18s/step - loss: 0.0019 - accuracy: 1.0000 - val_loss: 3.0736 - val_accuracy: 0.8636
# Epoch 5/10
# 1/1 [==============================] - 18s 18s/step - loss: 0.4620 - accuracy: 0.9828 - val_loss: 5.9451 - val_accuracy: 0.8182
# Epoch 6/10
# 1/1 [==============================] - 21s 21s/step - loss: 0.2193 - accuracy: 0.9828 - val_loss: 7.2778 - val_accuracy: 0.8182
# Epoch 7/10
# 1/1 [==============================] - 20s 20s/step - loss: 2.6228e-08 - accuracy: 1.0000 - val_loss: 8.4450 - val_accuracy: 0.8182
# Epoch 8/10
# 1/1 [==============================] - 19s 19s/step - loss: 3.3127e-08 - accuracy: 1.0000 - val_loss: 9.4671 - val_accuracy: 0.8182
# Epoch 9/10
# 1/1 [==============================] - 18s 18s/step - loss: 4.4000e-08 - accuracy: 1.0000 - val_loss: 10.3545 - val_accuracy: 0.8182
# Epoch 10/10
# 1/1 [==============================] - 20s 20s/step - loss: 6.8042e-08 - accuracy: 1.0000 - val_loss: 11.0988 - val_accuracy: 0.8182
# /home/prasad/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:1877: UserWarning: `Model.evaluate_generator` is deprecated and will be removed in a future version. Please use `Model.evaluate`, which supports generators.
#   warnings.warn('`Model.evaluate_generator` is deprecated and '
# > 81.818



################################################ transfer learning #####################################################

# import sys
# from matplotlib import pyplot
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.optimizers import SGD
# from keras.preprocessing.image import ImageDataGenerator
 
# # define cnn model
# def define_model():
# 	model = Sequential()
# 	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
# 	model.add(MaxPooling2D((2, 2)))
# 	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# 	model.add(MaxPooling2D((2, 2)))
# 	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# 	model.add(MaxPooling2D((2, 2)))
# 	model.add(Flatten())
# 	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dense(1, activation='sigmoid'))
# 	# compile model
# 	opt = SGD(lr=0.001, momentum=0.9)
# 	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
# 	return model
 
# # plot diagnostic learning curves
# def summarize_diagnostics(history):
# 	# plot loss
# 	pyplot.subplot(211)
# 	pyplot.title('Cross Entropy Loss')
# 	pyplot.plot(history.history['loss'], color='blue', label='train')
# 	pyplot.plot(history.history['val_loss'], color='orange', label='test')
# 	# plot accuracy
# 	pyplot.subplot(212)
# 	pyplot.title('Classification Accuracy')
# 	pyplot.plot(history.history['accuracy'], color='blue', label='train')
# 	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
# 	# save plot to file
# 	filename = sys.argv[0].split('/')[-1]
# 	pyplot.savefig(filename + '_plot.png')
# 	pyplot.close()
 
# # run the test harness for evaluating a model
# def run_test_harness():
# 	# define model
# 	model = define_model()
# 	# create data generators
# 	train_datagen = ImageDataGenerator(rescale=1.0/255.0,
# 		width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
# 	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
# 	# prepare iterators
# 	train_it = train_datagen.flow_from_directory('dataset_ant_vs_parrot_final/train/',
# 		class_mode='binary', batch_size=64, target_size=(200, 200))
# 	test_it = test_datagen.flow_from_directory('dataset_ant_vs_parrot_final/test/',
# 		class_mode='binary', batch_size=64, target_size=(200, 200))
# 	# fit model
# 	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
# 		validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=0)
# 	# evaluate model
# 	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
# 	print('> %.3f' % (acc * 100.0))
# 	# learning curves
# 	summarize_diagnostics(history)
 
# # entry point, run the test harness
# run_test_harness()
######################################### results for sequential #############################


# 2021-04-23 16:58:04.468159: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
# 2021-04-23 16:58:04.468262: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
# 2021-04-23 16:58:17.044281: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
# 2021-04-23 16:58:17.082625: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
# 2021-04-23 16:58:17.082669: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
# 2021-04-23 16:58:17.082709: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (prasad): /proc/driver/nvidia/version does not exist
# 2021-04-23 16:58:17.083090: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2021-04-23 16:58:17.083684: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
# Found 58 images belonging to 2 classes.
# Found 22 images belonging to 2 classes.
# /home/prasad/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:1844: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
#   warnings.warn('`Model.fit_generator` is deprecated and '
# 2021-04-23 16:58:18.284748: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
# 2021-04-23 16:58:18.428894: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 1800000000 Hz
# 2021-04-23 16:58:20.095841: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 296960000 exceeds 10% of free system memory.
# 2021-04-23 16:58:21.905756: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 296960000 exceeds 10% of free system memory.
# 2021-04-23 16:58:23.384439: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 296960000 exceeds 10% of free system memory.
# 2021-04-23 16:58:25.003008: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 296960000 exceeds 10% of free system memory.
# 2021-04-23 16:58:26.104349: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 296960000 exceeds 10% of free system memory.
# /home/prasad/.local/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:1877: UserWarning: `Model.evaluate_generator` is deprecated and will be removed in a future version. Please use `Model.evaluate`, which supports generators.
#   warnings.warn('`Model.evaluate_generator` is deprecated and '
# > 63.636

############################################################################################