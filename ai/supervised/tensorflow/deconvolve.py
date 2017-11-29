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
        
        self.weights = {'w': keras_layer.get_weights()[0],
                        'b': keras_layer.get_weights()[1]} 
        self.output = np.zeros(keras_layer.output_shape[1:], dtype=np.float32)
        self.k = keras_layer.kernel_size[0]
        
    def feedforward(self, x):
        self.output.fill(0)
        for i in range(self.output.shape[0]):
            for j in range(x.shape[0]):
                for k in range(self.output.shape[1]):
                    for l in range(self.output.shape[2]):
                        self.output[i, k, l] += \
                            (x[j, k:k+self.k, l:l+self.k]*self.weights['w'][:, :, j, i]).sum()
            self.output[i] += self.weights['b'][i]
            self.output[i][self.output[i, :, :] < 0] = 0 # Relu
        return self.output

class MaxPooling2D:
    def __init__(self, keras_layer):
        input_shape = keras_layer.output_shape[1:]
        output_shape = keras_layer.output_shape[1:]
        self.p = keras_layer.pool_size[0]        
        self.output = np.zeros((output_shape), dtype=np.float32)
        self.indices = np.zeros(list(output_shape)+[2], dtype=np.int32)
       
    def feedforward(self, x):
        for i in range(x.shape[0]):
            for j in range(0, x.shape[1]//2*2, self.p):
                for k in range(0, x.shape[2]//2*2, self.p):
                    slice_of_input = x[i, j:j+self.p, k:k+self.p]
                    max_value = slice_of_input.max()
                    self.output[i, j//self.p, k//self.p] = max_value
                    max_index = np.where(slice_of_input == max_value)
                    self.indices[i, j//self.p, k//self.p] = \
                        np.array([max_index[0][0], max_index[1][0]]) + np.array([j, k])
                    
        return self.output

class Flatten:
    def __init__(self, keras_layer):
        pass
        
    def feedforward(self, x):
        self.output  = x.flatten().reshape((1, -1))
        return self.output

class Dense:
    def __init__(self, keras_layer):
        
        self.weights = {'w': keras_layer.get_weights()[0],
                        'b': keras_layer.get_weights()[1]}
        self.activation = keras_layer.activation.__name__
        self.output = np.zeros(keras_layer.output_shape[1:], dtype=np.float32)
        
    def feedforward(self, x):
        self.output = x.dot(self.weights['w']) + self.weights['b']
        if self.activation is 'relu':
            self.output[self.output < 0] = 0
        elif self.activation is 'sigmoid':
            self.output = 1/(1+np.exp(-self.output))
        elif self.activation is 'softmax':
            z_exp = np.exp(self.output)
            self.output = z_exp / z_exp.sum()
        else:
            raise Exception('Invalid activation')
        return self.output

def unpool(x, indices, shape):
    output = np.zeros(shape, dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                jj, kk = indices[i, j, k]
                output[i, jj, kk] = x[i, j, k]
#        print(x[i])
#        print(output[i])
    return output

def deconv(x, weights):
    x[x < 0] = 0 # Relu
    output = np.zeros((weights.shape[2], 
                      x.shape[1] + weights.shape[0] - 1, 
                      x.shape[2] + weights.shape[1] - 1), dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(output.shape[0]):
            for k in range(x.shape[1]):
                for l in range(x.shape[2]):
                    output[j, k:k+weights.shape[0], l:l+weights.shape[1]] += \
                        (x[i, k, l] * weights[:, :, j, i].transpose())
    return output


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
              epochs=200,
              verbose=1,
              validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        model.save(model_file_name)
    
    return model, x_train, y_train
    

if __name__ == "__main__":
    
    keras_model = load_model('models/model.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    with h5py.File('../data/2017-11-19-04-18-11_2_3_100.h5', 'r') as f:
        frames = f['frames'][:]
    
#    keras_model, frames, labels = mnist_model()
    model = from_keras_model(keras_model)
    
#    for i in range(6):
#        frame = frames[i]
#            
#        outputs = []
#        outputs.append(frame)
#        for layer in model:
#            outputs.append(layer.feedforward(outputs[-1]))
#        print(keras_model.predict(frame.reshape((1,9,128,128))).flatten())
#        print(outputs[-1].flatten())
##        print(outputs[-1] == keras_model.predict(frame.reshape((1,1,28,28))))
#        
#        def plot(activations):
#            plt.figure(0)
#            n, h, w = activations.shape
#            n_cols = min(n, 8)
#            n_rows = n // n_cols
#            for i in range(n_rows):
#                for j in range(n_cols):
#                    plt.subplot2grid((n_rows, n_cols), (i,j))
#                    plt.imshow(activations[i*n_cols+j].reshape((h, w)))
#            plt.show()
#        
#        for i in range(1, 7, 2):
#            output = outputs[i]
#            print(output.shape)
#            plot(outputs[i])
#        

#    
    frame = frames[73]
#    label = labels[2]
    
    outputs = []
    outputs.append(frame)
    for layer in model:
#        print(outputs[-1])
        outputs.append(layer.feedforward(outputs[-1]))
#    print(keras_model.predict(frame.reshape((1,9,128,128))).flatten())
#    print(outputs[-1].flatten())
    
    idx_l = 0
    deconvolved_weights = np.zeros_like(model[idx_l].weights['w'])
    for idx in range(model[idx_l].weights['w'].shape[2]):
        for jdx in range(model[idx_l].weights['w'].shape[3]):
            
            deconvolved_weights.fill(0)
            deconvolved_weights[:, :, idx, jdx] = np.copy(model[idx_l].weights['w'][:, :, idx, jdx])
                        
#            deconv2 = deconv(model[2].output, deconvolved_weights)
#            unpool1 = unpool(deconv2, model[1].indices, model[0].output.shape)
#            deconv1 = deconv(unpool1, model[0].weights['w'])#.reshape((28, 28))
            deconv1 = deconv(model[0].output, deconvolved_weights)#.reshape((28, 28))
            
            def plot(activations):
                plt.figure(0)
                n, h, w = activations.shape
                n_cols = min(n, 3)
                n_rows = n // n_cols
                for i in range(n_rows):
                    for j in range(n_cols):
                        plt.subplot2grid((n_rows, n_cols), (j,i))
                        plt.imshow(activations[i*n_cols+j].reshape((h, w)))
                plt.show()
            plot(deconv1)
            print('----------------------------------------------------------------------------------')
            
#            f, ax = plt.subplots(1, 3)
#            ax[0].imshow(deconv1[0:3].transpose((1,2,0)))
#            ax[1].imshow(deconv1[3:6].transpose((1,2,0)))
#            ax[2].imshow(deconv1[6:9].transpose((1,2,0)))
#            plt.show()
        
#            plt.imshow(deconv1) 
#            plt.title(str(idx) + ' ' + str(jdx))
#            plt.show()
#    
    
    
    
    
    