import cv2
image = cv2.imread('/Users/xiangtiange1/Desktop/cbir/pathology_image/Training_data/Benign/t9.tif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(224,224))
#use cv2 to read in image and resize to fit in VGG

def get_map(arg):
	"""
	Helper method to get index map
	arg: output BEFORE max pooling
	return: index map
	"""
    index_map = np.zeros((arg.shape[3],arg.shape[1]//2, arg.shape[2]//2))
    for idx in range(arg.shape[3]):
        for row in range(arg.shape[1]//2):
            for col in range(arg.shape[2]//2):
                index_map[idx][row][col] = np.argmax(arg[0,:,:,idx][row*2:row*2+2,col*2:col*2+2])
    print(index_map.shape)
    return index_map

def unpooling(index, value):
	"""
	Helper method to unpool
	index: index map from get_map
	value: incoming feature map for unpooling
	return: unpooled feature map
	"""
    holders = np.zeros((1, index.shape[1] * 2, index.shape[2] * 2, index.shape[0]))
    for idx in range(index.shape[0]):
        for row in range(index.shape[1]):
            for col in range(index.shape[2]):
                if index[idx,row,col] == 0:
                    holders[0,row*2,col*2,idx] = value[0,row,col,idx]
                elif index[idx,row,col] == 1:
                    holders[0,row*2 + 1,col*2,idx] = value[0,row,col,idx]
                elif index[idx,row,col] == 2:
                    holders[0,row*2,col*2 + 1,idx] = value[0,row,col,idx]
                elif index[idx,row,col] == 3:
                    holders[0,row*2 + 1,col*2 + 1,idx] = value[0,row,col,idx]
    print(holders.shape)
    return holders

#---------------dependencies---------------
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model

from keras.utils.np_utils import *
from keras.preprocessing import image
#-----------construct deconv blocks---------
'''
unpool the feature map between every adjcant block.
to capture a feature map of a particular block, set the 3rd parameters of input_tenosr shape to be 1.
e.g. fetch one feature map from block 4, set input_tensor4 = Input(shape = (img_size/8, img_size/8, 512 --> 1))
Only the captured block input_tensor should be set to 1, all others maintain the same shape.
'''
img_size = 224

input_tensor5 = Input(shape = (img_size/16, img_size/16, 512))
# x = Activation("relu")(input_tensor5)
x = Conv2DTranspose(512,(3,3), activation='relu', padding='same', name='t13')(input_tensor5)
#x = Activation("relu")(x)
x = Conv2DTranspose(512,(3,3), activation='relu', padding='same', name='t12')(x)
#x = Activation("relu")(x)
x = Conv2DTranspose(512,(3,3), activation='relu', padding='same', name='t11')(x)
block5_model = Model(input_tensor5, x)

input_tensor4 = Input(shape = (img_size/8, img_size/8, 512))
#x = Activation("relu")(input_tensor4)
x = Conv2DTranspose(512,(3,3), activation='relu', padding='same', name='t10')(input_tensor4)
#x = Activation("relu")(x)
x = Conv2DTranspose(512,(3,3), activation='relu', padding='same', name='t9')(x)
#x = Activation("relu")(x)
x = Conv2DTranspose(256,(3,3), activation='relu', padding='same', name='t8')(x)
block4_model = Model(input_tensor4, x)

input_tensor3 = Input(shape = (img_size/4, img_size/4, 256))
#x = Activation("relu")(input_tensor3)
x = Conv2DTranspose(256,(3,3), activation='relu', padding='same', name='t7')(input_tensor3)
#x = Activation("relu")(x)
x = Conv2DTranspose(256,(3,3), activation='relu', padding='same', name='t6')(x)
#x = Activation("relu")(x)
x = Conv2DTranspose(128,(3,3), activation='relu', padding='same', name='t5')(x)
block3_model = Model(input_tensor3, x)

input_tensor2 = Input(shape = (img_size/2, img_size/2, 128))
#x = Activation("relu")(input_tensor2)
x = Conv2DTranspose(128,(3,3), activation='relu', padding='same', name='t4')(input_tensor2)
#x = Activation("relu")(x)
x = Conv2DTranspose(64,(3,3), activation='relu', padding='same', name='t3')(x)
block2_model = Model(input_tensor2, x)

input_tensor1 = Input(shape = (img_size, img_size, 64))
#x = Activation("relu")(input_tensor1)
x = Conv2DTranspose(64,(3,3), activation='relu', padding='same', name='t2')(input_tensor1)
#x = Activation("relu")(x)
x = Conv2DTranspose(3,(3,3), activation='relu', padding='same', name='t1')(x)
#x = Activation("relu")(x)
block1_model = Model(input_tensor1, x)


#-----------find the most predominant feature map----
fetch = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2d_52').output)
img_in = fetch.predict([[image]])
mmax = 0
marg = -1 #saved as index to select the same filter while setting weights
for i in range(img_in.shape[3]):
    if np.sum(img_in[0,:,:,i]) > mmax:
        marg = i
        mmax = np.sum(img_in[0,:,:,i])
print(mmax, marg)


#-----------set weights---------
'''
weights[0]: filters (maintain the same)
weights[1]: bias (all set to 0)
uncomment the code in corresponding block, which is supposed to be consistent with the precious construct section.
All other blocks should leave the commented code commented.
'''
weights = base_model.get_layer('conv2d_52').get_weights()
weights[1] = weights[1] * 0
# rep = weights[0][:,:,:,marg]
# rep = np.expand_dims(rep, axis=3)
# print(rep.shape)
# weights[0] = rep
block5_model.get_layer('t13').set_weights(weights)

weights = base_model.get_layer('conv2d_51').get_weights()
weights[1] = weights[1] * 0
block5_model.get_layer('t12').set_weights(weights)

weights = base_model.get_layer('conv2d_50').get_weights()
weights[1] = weights[1] * 0
block5_model.get_layer('t11').set_weights(weights)

weights = base_model.get_layer('conv2d_49').get_weights()
weights[1] = weights[1] * 0
# rep = weights[0][:,:,:,marg]
# rep = np.expand_dims(rep, axis=3)
# print(rep.shape)
# weights[0] = rep
block4_model.get_layer('t10').set_weights(weights)

weights = base_model.get_layer('conv2d_48').get_weights()
weights[1] = weights[1] * 0
block4_model.get_layer('t9').set_weights(weights)

weights = base_model.get_layer('conv2d_47').get_weights()
weights[1] = weights[1][0:256] * 0
block4_model.get_layer('t8').set_weights(weights)

weights = base_model.get_layer('conv2d_46').get_weights()
weights[1] = weights[1] * 0
# rep = weights[0][:,:,:,marg]
# rep = np.expand_dims(rep, axis=3)
# print(rep.shape)
# weights[0] = rep
block3_model.get_layer('t7').set_weights(weights)

weights = base_model.get_layer('conv2d_45').get_weights()
weights[1] = weights[1] * 0

block3_model.get_layer('t6').set_weights(weights)

weights = base_model.get_layer('conv2d_44').get_weights()
weights[1] = weights[1][0:128] * 0
block3_model.get_layer('t5').set_weights(weights)

weights = base_model.get_layer('conv2d_43').get_weights()
weights[1] = weights[1] * 0
# rep = weights[0][:,:,:,marg]
# rep = np.expand_dims(rep, axis=3)
# print(rep.shape)
# weights[0] = rep
block2_model.get_layer('t4').set_weights(weights)

weights = base_model.get_layer('conv2d_42').get_weights()
weights[1] = weights[1][0:64] * 0
block2_model.get_layer('t3').set_weights(weights)

weights = base_model.get_layer('conv2d_41').get_weights()
weights[1] = weights[1] * 0
# rep = weights[0][:,:,:,marg]
# rep = np.expand_dims(rep, axis=3)
# print(rep.shape)
# weights[0] = rep
block1_model.get_layer('t2').set_weights(weights)

weights = base_model.get_layer('conv2d_40').get_weights()
weights[1] = weights[1][0:3] * 0
block1_model.get_layer('t1').set_weights(weights)
#-----------set weights ends---------

#-----------process---------
'''
general work flow: feature map => (deconv => relu => unpooling) * conv_layer_number => output
ready_pools: get the most recent feature map BEFORE the upcoming pooling layer, so that we can record the max indices.
The process MUST ALWAYS start with phase1, then insert phase1 to the blocks you whant.
phase6 is the final result matrix.
'''
phase1 = np.expand_dims(fetch.predict([[image]])[:,:,:,marg],axis=3)

phase2 = block5_model.predict([phase1])#default start point if you want to work with feature map from block 5

fetch = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2d_49').output)
ready_pool = fetch.predict([[image]])

index_map = get_map(ready_pool)
unpooled = unpooling(index_map, phase2)

phase3 = block4_model.predict([unpooled])
#start from here, set unpooled --> phase1 if you want to work with feature map from block 4 

fetch = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2d_46').output)
ready_pool2 = fetch.predict([[image]])

index_map2 = get_map(ready_pool2)
unpooled2 = unpooling(index_map2, phase3)

phase4 = block3_model.predict([unpooled2])
#start from here, set unpooled2 --> phase1 if you want to work with feature map from block 3

fetch = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2d_43').output)
ready_pool3 = fetch.predict([[image]])

index_map3 = get_map(ready_pool3)
unpooled3 = unpooling(index_map3, phase4)

phase5 = block2_model.predict([unpooled3])
#start from here, set unpooled3 --> phase1 if you want to work with feature map from block 2

fetch = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2d_41').output)
ready_pool4 = fetch.predict([[image]])

index_map4 = get_map(ready_pool4)
unpooled4 = unpooling(index_map4, phase5)

phase6 = block1_model.predict([unpooled4])
#start from here, set unpooled4 --> phase1 if you want to work with feature map from block 1

#-----------show output---------
'''
rerange the output to 0-1
'''
import numpy as np
from matplotlib import pyplot as plt
for idx in range(3):
    print(np.min(phase6[0,:,:,idx]), np.max(phase6[0,:,:,idx]), np.mean(phase6[0,:,:,idx]))
    if np.min(phase6[0,:,:,idx]) < 0:
        phase6[0,:,:,idx] = phase6[0,:,:,idx] + -1 * np.min(phase6[0,:,:,idx])
    if np.max(phase6[0,:,:,idx]) != 0:
        phase6[0,:,:,idx] = phase6[0,:,:,idx] /np.max(phase6[0,:,:,idx])
    
plt.imshow(phase6[0,:,:,:])
plt.show()
#plt.imsave('result.tif',phase6[0,:,:,:])