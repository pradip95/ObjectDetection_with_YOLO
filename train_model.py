"""
Created on Monday, January 24
@author : Pradip Mehta
"""

import pickle
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Input, LeakyReLU, Add, Concatenate
from tensorflow import data
import data_loader
import loss


def convolution(input_layer, params, down_sample=False, activate=True, bn=True):
    # params = (filter_shape) = (f_w, f_h, f_ch, n_f)

    padding = 'same'

    if down_sample:
        strides = 2

    else:
        strides = 1

    conv_output = Conv2D(filters=params[-1], kernel_size=params[0], strides=strides, padding=padding,
                         kernel_initializer='he_normal',
                         bias_initializer=tf.random_normal_initializer(0.))(input_layer)
    if bn:
        conv_output = BatchNormalization()(conv_output)

    if activate:
        conv_output = LeakyReLU()(conv_output)

    return conv_output


def convolution_transpose(input_layer, params, activate=True, bn=True):
    # params = (filter_shape) = (f_w, f_h, f_ch, n_f)

    padding = 'same'
    strides = 2

    conv_output = Conv2DTranspose(filters=params[-1], kernel_size=params[0], strides=strides, padding=padding,
                                  kernel_initializer='he_normal')(input_layer)

    if activate:
        conv_output = LeakyReLU()(conv_output)

    if bn:
        conv_output = BatchNormalization()(conv_output)

    return conv_output


def residual_block(input_layer, input_channel, filter_ch_1, filter_ch_2, repeat_block):
    # inputs to residual block
    # input_layer ---> input dimensions
    # input_channel ---> channels of previous layer
    # filter_ch_1 ---> number of filters for 1st convolution, 1x1 kernel
    # filter_ch_2 ---> number of filters for 2nd convolution, 3x3 kernel
    # skip_add ---> for adding input_layer to residual output after skipping some convolution layers

    skip_add = input_layer
    for repeat in range(repeat_block):
        res_conv = convolution(input_layer, (1, 1, input_channel, filter_ch_1), activate=False, bn=False)
        res_conv = convolution(res_conv, (3, 3, filter_ch_1, filter_ch_2), activate=False, bn=False)

        residual_output = Add()([res_conv, skip_add])

        residual_output = LeakyReLU()(residual_output)

        return residual_output


def darknet_block(input_layer):
    # first two convolution layers
    input_layer = convolution(input_layer, (3, 3, 3, 32))  # params = (filter_shape) = (f_w, f_h, f_ch, n_f)
    input_layer = convolution(input_layer, (3, 3, 32, 64), down_sample=True)  # reducing spatial resolution by 1/2

    # first residual block x 1 (3-4 convolution layers)
    # input_layer, input_channel, filter_ch_1, filter_ch_2, repeat_block
    input_layer = residual_block(input_layer, 64, 32, 64, 1)

    # 5th convolution layer
    input_layer = convolution(input_layer, (3, 3, 64, 128), down_sample=True)  # reducing spatial resolution by 1/4

    # second residual block x 2 (6-9 convolution layers)
    input_layer = residual_block(input_layer, 128, 64, 128, 2)

    # 10th convolution layer
    input_layer = convolution(input_layer, (3, 3, 128, 256), down_sample=True)  # reducing spatial resolution by 1/8

    # third residual block x 8 (11-26 convolution layers)
    input_layer = residual_block(input_layer, 256, 128, 256, 8)

    # first connection to network head. 256 filters, 1/8 size (52 x 52)
    connect1 = input_layer

    # 27th convolution layer
    input_layer = convolution(input_layer, (3, 3, 256, 512), down_sample=True)  # reducing spatial resolution by 1/16

    # fourth residual block x 8 (28-43 convolution layers)
    input_layer = residual_block(input_layer, 512, 256, 512, 8)

    # second connection to network head. 512 filters, 1/16 size (26 x 26)
    connect2 = input_layer

    # 44th convolution layer
    input_layer = convolution(input_layer, (3, 3, 512, 1024), down_sample=True)  # reducing spatial resolution by 1/32

    # fifth residual block x 4 (43-51 convolution layers)
    input_layer = residual_block(input_layer, 1024, 512, 1024, 4)

    # third connection to network head. 1024 filters, 1/32 size (13 x 13)
    connect3 = input_layer
    # print(connect1, connect2, connect3)
    return connect1, connect2, connect3


def yolo_layers(input_shape, NUM_CLASSES):
    num_anchors = 1
    inputs = Input(shape=input_shape)

    connect1, connect2, connect3 = darknet_block(inputs)

    scale_3 = convolution(connect3, (1, 1, 1024, 512))
    scale_3 = convolution(scale_3, (3, 3, 512, 1024))
    scale_3 = convolution(scale_3, (1, 1, 1024, 512))
    scale_3 = convolution(scale_3, (3, 3, 512, 1024))
    scale_3 = convolution(scale_3, (1, 1, 1024, 512))

    scale_3out = convolution(scale_3, (3, 3, 512, 1024))
    scale_3out = convolution(scale_3out, (1, 1, 1024, num_anchors * (NUM_CLASSES + 5)), activate=False)

    in_scale_2 = convolution(scale_3, (1, 1, 512, 256))
    in_scale_2 = convolution_transpose(in_scale_2, (1, 1, 512, 256))  # up-sample
    in_scale_2 = Concatenate()([in_scale_2, connect2])

    scale_2 = convolution(in_scale_2, (1, 1, 512, 256))
    scale_2 = convolution(scale_2, (3, 3, 256, 512))
    scale_2 = convolution(scale_2, (1, 1, 512, 256))
    scale_2 = convolution(scale_2, (3, 3, 256, 512))
    scale_2 = convolution(scale_2, (1, 1, 512, 256))

    scale_2out = convolution(scale_2, (3, 3, 256, 512))
    scale_2out = convolution(scale_2out, (1, 1, 512, num_anchors * (NUM_CLASSES + 5)), activate=False)

    in_scale_1 = convolution(scale_2, (1, 1, 256, 128))
    in_scale_1 = convolution_transpose(in_scale_1, (1, 1, 256, 128))  # up-sample
    in_scale_1 = Concatenate()([in_scale_1, connect1])

    scale_1 = convolution(in_scale_1, (1, 1, 256, 128))
    scale_1 = convolution(scale_1, (3, 3, 128, 256))
    scale_1 = convolution(scale_1, (1, 1, 256, 128))
    scale_1 = convolution(scale_1, (3, 3, 128, 256))
    scale_1 = convolution(scale_1, (1, 1, 256, 128))

    scale_1out = convolution(scale_1, (3, 3, 128, 256))
    scale_1out = convolution(scale_1out, (1, 1, 256, num_anchors * (NUM_CLASSES + 5)), activate=False, bn=False)

    yolo_model = Model(inputs=inputs, outputs=[scale_3out])  # scale_2out 24,16 , scale_3out 12, 8
    print(scale_1out)
    print(scale_2out)
    print(scale_3out)
    return yolo_model


# number of classes
NUM_CLASS = 4

# for loading new dataset
load_data = data_loader.YOLODataset("data/train_dataset/", "data/train_dataset/",
                                    [384, 256], S=[[12, 8], [24, 16], [48, 32]], S_index=0, C=4)
x_train = load_data.x_dataset_loader()
y_train = load_data.y_dataset_loader()

load_data = data_loader.YOLODataset("data/validation_dataset/", "data/validation_dataset/",
                                    [384, 256], S=[[12, 8], [24, 16], [48, 32]], S_index=0, C=4)
x_valid = load_data.x_dataset_loader()
y_valid = load_data.y_dataset_loader()

'''
# load pickled image dataset and labels for training
x_train = open("data/pickled_data/x_train.pickle", "rb")
x_train = pickle.load(x_train)
y_train = open("data/pickled_data/y_train.pickle", "rb")
y_train = pickle.load(y_train)

# load pickled image dataset and labels for testing
x_test = open("data/pickled_data/x_test.pickle", "rb")
x_test = pickle.load(x_test)
y_test = open("data/pickled_data/y_test.pickle", "rb")
y_test = pickle.load(y_test)
'''
# input dimension of image to cnn
input_img_dim = x_train.shape[1:]

model = yolo_layers(input_img_dim, NUM_CLASS)
model.summary()

# custom loss function
loss_func = loss.LossFunction()
losses = loss_func.loss_fn

# number of examples to be passed to cnn for training and validation
if len(x_train) > 1000:
    # total number of examples in Test dataset are 2826
    x_train = x_train[500:]  # change number of examples for training x dataset here
    y_train = y_train[500:]  # change number of examples for training y dataset here

    # total number of examples in Test dataset are 450
    x_test = x_valid[30:]  # change number of examples for validation x dataset here
    y_test = y_valid[30:]  # change number of examples for validation y dataset here
    print('Training images found : ', len(x_train))
    print('Training labels found : ', len(y_train))
    print('Validation dataset examples found : ', len(x_test))


# train and save the model
model.compile(loss=losses, optimizer=Adam(learning_rate=0.001))
hist = model.fit(x_train, y_train, batch_size=2, epochs=150, validation_data=(x_valid, y_valid), verbose=2)
model.save('trained_models/yolo.model10')  # change the version of model trained

# plotting the losses
plot_hist = pd.DataFrame(hist.history)
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Evaluation')
plt.plot(plot_hist['loss'], label='Training Error')
plt.plot(plot_hist['val_loss'], label='Validation Error')
plt.legend()
plt.savefig('trained_models/Loss_plot_model10.png')  # change the version of model trained
plt.show()
