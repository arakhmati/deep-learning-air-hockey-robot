import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.layers
from model import fmeasure, recall, precision
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
        self.layer_output = np.zeros((input_shape[1] - self.weight.shape[0] + 1, 
                                      input_shape[2] - self.weight.shape[1] + 1, 
                                      self.weight.shape[3]), dtype=np.float32)
        
        self.n_in  = input_shape[3]
        self.n_out = self.layer_output.shape[2]
        
        self.height_out = input_shape[1] - self.weight.shape[0] + 1
        self.width_out  = input_shape[2] - self.weight.shape[1] + 1
        
        self.kernel_dim = self.weight.shape[0]
        
        self.temp = np.zeros((self.kernel_dim, self.kernel_dim), dtype = np.float32)
        
    def feedforward(self, layer_input):
        self.layer_output.fill(0)
        
        
        
        for i in range(self.n_out):
            for j in range(self.n_in):
                for k in range(self.height_out):
                    for l in range(self.width_out):
                        self.temp = (layer_input[k:k+self.kernel_dim, l:l+self.kernel_dim, j]*self.weight[:, :, j, i])
                        self.layer_output[k, l, i] += self.temp.sum()
            self.layer_output[:, :, i] += self.bias[i]
            self.layer_output[:, :, i][self.layer_output[:, :, i] < 0] = 0 # Relu
        return self.layer_output

class MaxPooling2D:
    def __init__(self, keras_layer):
        input_shape = keras_layer.input_shape
        self.pool_size = keras_layer.pool_size[0]
        
        self.layer_output = np.zeros((input_shape[1]//self.pool_size, 
                                      input_shape[2]//self.pool_size, 
                                      input_shape[3]), dtype=np.float32)
        self.indices = np.zeros((input_shape[1]//self.pool_size, 
                                 input_shape[2]//self.pool_size, 
                                 input_shape[3], 2), dtype=np.int32)
       
    def feedforward(self, layer_input):
        for i in range(self.layer_output.shape[2]):
            for j in range(self.layer_output.shape[0]):
                for k in range(self.layer_output.shape[1]):
                    slice_of_input = layer_input[j*self.pool_size:j*self.pool_size+self.pool_size, 
                                                 k*self.pool_size:k*self.pool_size+self.pool_size, i]
                    max_index = np.where(slice_of_input == slice_of_input.max())
                    self.indices[j, k, i] = np.array([max_index[0][0], max_index[1][0]]) + \
                                    np.array([j*self.pool_size, k*self.pool_size])
                    self.layer_output[j, k ,i] = slice_of_input.max()
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
    for i in range(layer_input.shape[2]):
        for j in range(layer_input.shape[0]):
            for k in range(layer_input.shape[1]):
                layer_output[:,:,i][indices[j, k, i]] = layer_input[j, k, i]
    return layer_output

def deconv(layer_input, weight, bias):
    layer_input[:, :, :][layer_input[:, :, :] < 0] = 0 # Relu
    layer_output = np.zeros((layer_input.shape[0] + weight.shape[0] - 1, layer_input.shape[1] + weight.shape[1] - 1, 
                            weight.shape[2]), dtype=np.float32)
    for i in range(layer_input.shape[2]):
        for j in range(layer_output.shape[2]):
            for k in range(layer_output.shape[0] - weight.shape[0] + 1):
                for l in range(layer_output.shape[1] - weight.shape[1] + 1):
                    layer_output[k:k+weight.shape[0], l:l+weight.shape[1], j] += (layer_input[k, l, i] * weight[:, :, j, i].transpose())
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
    

if __name__ == "__main__":
    
    keras_model = load_model('conv.h5', {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
    model = from_keras_model(keras_model)
    
    with h5py.File('data_10.h5', 'r') as f:
        frames = f['frames'][:]
        
    frame = frames[9]
    
    out = np.copy(frame)
    for layer in model:
#        tic = time.time()
        out = layer.feedforward(out)
        
#    tic = time.time()
#    keras_out = keras_model.predict(frame.reshape((1,128,128,3)))
#    toc = time.time()
#    print(toc - tic)
    
    deconvolved_weights = np.copy(model[4].weight)
    for idx in range(model[4].weight.shape[2]):
        for jdx in range(model[4].weight.shape[3]):
            
            deconvolved_weights.fill(0)
            deconvolved_weights[:, :, idx, jdx] = np.copy(model[4].weight[:, :, idx, jdx])
                        
            deconv3 = deconv(model[4].layer_output, deconvolved_weights, model[4].bias)
            unpool2 = unpool(deconv3, model[3].indices, model[2].layer_output.shape)
            deconv2 = deconv(unpool2, model[2].weight, model[2].bias)
            unpool1 = unpool(deconv2, model[1].indices, model[0].layer_output.shape)
            deconv1 = deconv(unpool1, model[0].weight, model[0].bias)
            
            plt.imshow(deconv1) 
            plt.title(str(idx) + ' ' + str(jdx))
            plt.show()
#    
    
    
    
    
    