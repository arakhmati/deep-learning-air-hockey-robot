import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.layers
from model import fmeasure, recall, precision

from keras.datasets import mnist
'''
Decide which filter activation you want to visualize. Pass the image forward through the conv net, 
up to and including the layer where your chosen activation is.

Zero out all filter activations (channels) in the last layer except the one you want to visualize.

Now go back to image space, but through the deconv net. 
For this, the authors propose 'inverse' operations of the three common operations seen in conv nets.
Unpooling (see also bottom part of Figure 1): 
    Max pooling cannot be exactly inverted. 
    So the authors propose to remember the position of the max lower layer activation in 'switch variables'. 
    While going back, the activation from the upper layer is copy-pasted to the position pointed to by the switch variable, 
    and all other lower layer activations are set to zero. Note that different images will produce different patterns of activations, 
    so the values of the switch variables will change according to image.
ReLU: The inverse of the ReLU function is the ReLU function. 
    It sounds a bit odd, but the authors' argument is that since convolution is applied to rectified activations in the forward pass,
    deconvolution should also be applied to rectified reconstructions in the backward pass.
Deconvolution: This uses the same filters are the corresponding conv layer; 
    the only difference is that they are flipped horizontally and vertically.

Follow these three steps till you reach the image layer. 
The pattern that emerges in the image layer is the discriminative pattern that the selected activation is sensitive to. 
These are the greyish patches shown in Figure 2 in the paper.

The real-world image patches shown in Figure 2 besides the greyish patches are just crops of the input image, 
made by the receptive field of the chosen activation. 
'''

class Conv2D:
    def __init__(self, keras_layer):
        weights = keras_layer.get_weights()
        input_shape = keras_layer.input_shape

        self.weight = weights[0]
        self.bias   = weights[1]
        self.layer_output = np.zeros((self.weight.shape[3],
                                      input_shape[2] - self.weight.shape[0] + 1, 
                                      input_shape[3] - self.weight.shape[1] + 1), dtype=np.float32)
        
        self.n_in  = input_shape[1]
        self.n_out = self.layer_output.shape[0]
        
        self.height_out = input_shape[2] - self.weight.shape[0] + 1
        self.width_out  = input_shape[3] - self.weight.shape[1] + 1
        
        self.kernel_dim = self.weight.shape[0]
        
    def feedforward(self, layer_input):
        self.layer_output.fill(0)
        
        
#        print(self.weight.shape)
#        print(self.layer_output.shape)
        
        
        for i in range(self.n_out):
            for j in range(self.n_in):
                for k in range(self.height_out):
                    for l in range(self.width_out):
                        temp = layer_input[j, k:k+self.kernel_dim, l:l+self.kernel_dim]*self.weight[:, :, j, i]
                        self.layer_output[i, k, l] += temp.sum()
            self.layer_output[i, :, :] += self.bias[i]
            self.layer_output[i, :, :][self.layer_output[i, :, :] < 0] = 0 # Relu
        return self.layer_output

class MaxPooling2D:
    def __init__(self, keras_layer):
        input_shape = keras_layer.input_shape
        self.pool_size = keras_layer.pool_size[0]
        
        self.layer_output = np.zeros((input_shape[1], 
                                      input_shape[2]//self.pool_size, 
                                      input_shape[3]//self.pool_size), dtype=np.float32)
        self.indices = np.zeros((input_shape[1], 
                                 input_shape[2]//self.pool_size, 
                                 input_shape[3]//self.pool_size, 2), dtype=np.int32)
       
    def feedforward(self, layer_input):
        for i in range(self.layer_output.shape[0]):
            for j in range(self.layer_output.shape[1]):
                for k in range(self.layer_output.shape[2]):
                    slice_of_input = layer_input[i, j*self.pool_size:j*self.pool_size+self.pool_size, 
                                                 k*self.pool_size:k*self.pool_size+self.pool_size]
                    max_index = np.where(slice_of_input == slice_of_input.max())
                    self.indices[i, j, k] = np.array([max_index[0][0], max_index[1][0]]) + \
                                    np.array([j*self.pool_size, k*self.pool_size])
                    self.layer_output[i, j, k] = slice_of_input.max()
        return self.layer_output

class Flatten:
    def __init__(self, keras_layer):
        self.layer_output = np.zeros(keras_layer.output_shape[1], dtype=np.float32)
        
    def feedforward(self, layer_input):
        self.layer_output  = layer_input.flatten().reshape((1, -1))
        return self.layer_output

class Dense:
    def __init__(self, keras_layer):
        weights = keras_layer.get_weights()
        self.weight = weights[0]
        self.bias = weights[1]
        self.activation = keras_layer.activation.__name__
        self.layer_output = np.zeros((self.weight.shape[1]), dtype=np.float32)
        
    def feedforward(self, layer_input):
        self.layer_output = layer_input.dot(self.weight) + self.bias
        if self.activation is 'relu':
            self.layer_output[self.layer_output < 0] = 0
        elif self.activation is 'sigmoid':
            self.layer_output = 1/(1+np.exp(-self.layer_output))
        elif self.activation is 'softmax':
            z_exp = np.exp(self.layer_output)
            self.layer_output = z_exp / z_exp.sum()
        else:
            raise Exception('Invalid activation')
        return self.layer_output

def unpool(layer_input, indices, shape):
    layer_output = np.zeros(shape, dtype=np.float32)
    for i in range(layer_input.shape[0]):
        for j in range(layer_input.shape[1]):
            for k in range(layer_input.shape[2]):
                layer_output[i,:,:][indices[i, j, k]] = layer_input[i, j, k]
    return layer_output

def deconv(layer_input, weight, bias):
    layer_input[:][layer_input < 0] = 0 # Relu
    layer_output = np.zeros((weight.shape[2], 
                             layer_input.shape[1] + weight.shape[0] - 1, 
                             layer_input.shape[2] + weight.shape[1] - 1), dtype=np.float32)
    for i in range(layer_input.shape[0]):
        for j in range(layer_output.shape[0]):
            for k in range(layer_input.shape[1]):
                for l in range(layer_input.shape[2]):
                    layer_output[j, k:k+weight.shape[0], l:l+weight.shape[1]] += (layer_input[i, k, l] * weight[:, :, j, i].transpose())
    return layer_output


def from_keras_model(keras_model):
    
    model = []
    
    for keras_layer in keras_model.layers:
        if isinstance(keras_layer, keras.layers.Conv2D):
            model.append(Conv2D(keras_layer))
        if isinstance(keras_layer, keras.layers.MaxPooling2D):
            model.append(MaxPooling2D(keras_layer))
        if isinstance(keras_layer, keras.layers.Flatten):
            model.append(Flatten(keras_layer))
        if isinstance(keras_layer, keras.layers.Dense):
            model.append(Dense(keras_layer))
    return model

def mnist_model():
    
    model_file_name = 'mnist_model.h5'
    
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    
    if os.path.exists(model_file_name):
        print('Model already exists. Loading...')
        model = load_model(model_file_name)
    else:
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
        model = Sequential([
                    Conv2D(16, 5, activation='relu', name='conv1', 
                           data_format='channels_first',
                           input_shape=(1, 28, 28)),
                    MaxPooling2D(2, name='pool1', data_format='channels_first'),
                    Conv2D(32, 5, activation='relu', name='conv2', 
                           data_format='channels_first'),
                    MaxPooling2D(2, name='pool2', data_format='channels_first'),
                    Dropout(0.4, name='conv4_dropout'),
                    Flatten(name='flatten'),
                    Dense(128, activation='relu', name='dense1'),
                    Dropout(0.3, name='dense1_dropout'),
                    Dense(64, activation='relu', name='dense2'),
                    Dense(10,  activation='softmax', name='out')
                    ])
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        model.fit(x_train, y_train,
              batch_size=128,
              epochs=20,
              verbose=1,
              validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        model.save(model_file_name)
    
    return model, x_train
    

if __name__ == "__main__":
    
#    keras_model = load_model('model.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
#    with h5py.File('../data/2017-11-09-00-21-14_10.h5', 'r') as f:
#        frames = f['frames'][:]
    
    keras_model, frames = mnist_model()
#    frame = frames[0].reshape((1,1,28,28))
#    print(keras_model.predict(frame.reshape((1,1,28,28))))
#    
#    import tensorflow as tf
#    config = tf.ConfigProto(device_count = {'GPU': 0})
#    with tf.Session(config=config) as sess:
#        print(tf.get_default_graph().get_operations())
#        
#        init = tf.constant(keras_model.layers[0].get_weights()[0])
#        tf.get_variable('conv1_kernel:0', initializer=init)
#        init = tf.constant(keras_model.layers[0].get_weights()[1])
#        tf.get_variable('conv1_bias:0', initializer=init)
#        
#        
#        
#        sess.run(keras_model.layers[0], feed_dict={'conv1_input:0': frame})
#        
#        
    model = from_keras_model(keras_model)
    
    for i in range(50):
        frame = frames[i]
            
        outputs = []
        outputs.append(frame)
        for layer in model:
            outputs.append(layer.feedforward(outputs[-1]))
        print(keras_model.predict(frame.reshape((1,1,28,28))).flatten())
        print(outputs[-1].flatten())
        print()
        plt.imshow(frame.reshape((28,28))) 
        plt.title(np.argmax(outputs[-1].flatten()))
        plt.show()
#        print(outputs[-1] == keras_model.predict(frame.reshape((1,1,28,28))))
        
        def plot(activations):
            plt.figure(0)
            n, h, w = activations.shape
            n_cols = min(n, 8)
            n_rows = n // n_cols
            for i in range(n_rows):
                for j in range(n_cols):
                    plt.subplot2grid((n_rows, n_cols), (i,j))
                    plt.imshow(activations[i*n_cols+j].reshape((h, w)))
            plt.show()
        
#        for i in range(5):
#            plot(outputs[i])
        
#    tic = time.time()
#    keras_out = keras_model.predict(frame.reshape((1,128,128,3)))
#    toc = time.time()
#    print(toc - tic)
    
    deconvolved_weights = np.copy(model[2].weight)
    print(model[2].weight.shape)
    for idx in range(model[2].weight.shape[2]):
        for jdx in range(model[2].weight.shape[3]):
            
            deconvolved_weights.fill(0)
            deconvolved_weights[:, :, idx, jdx] = np.copy(model[2].weight[:, :, idx, jdx])
                        
            deconv2 = deconv(model[2].layer_output, model[2].weight, model[2].bias)
#            plt.imshow(deconv2[0].reshape((12,12))) 
#            plt.title(str(idx) + ' ' + str(jdx))
#            plt.show()
            unpool1 = unpool(deconv2, model[1].indices, model[0].layer_output.shape)
#            print(unpool1.shape)
            deconv1 = deconv(unpool1, model[0].weight, model[0].bias).reshape((28, 28))
#            print(deconv1.shape)
            
            plt.imshow(deconv1) 
            plt.title(str(idx) + ' ' + str(jdx))
            plt.show()
##    
    
    
    
    
    