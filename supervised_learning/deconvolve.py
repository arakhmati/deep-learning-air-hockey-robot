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
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.models import load_model

if __name__ == "__main__":
    
    model = load_model('bottom_ai_model.h5')
#    model = build_model()
    
    with h5py.File('frame.h5', 'r') as f:
        frame = f['frame'][:]
    
    weights = np.copy(model.get_weights())
    
    
    def conv_layer(layer_input, weight, bias):
        layer_output = np.zeros((layer_input.shape[0] - weight.shape[0] + 1, layer_input.shape[1] - weight.shape[1] + 1, 
                                weight.shape[3]), dtype=np.float32)
        for i in range(layer_output.shape[2]):
            for j in range(layer_input.shape[2]):
                for k in range(layer_input.shape[0] - weight.shape[0] + 1):
                    for l in range(layer_input.shape[1] - weight.shape[1] + 1):
                        layer_output[k, l, i] += (layer_input[k:k+weight.shape[0], l:l+weight.shape[1], j]*weight[:, :, j, i]).sum()
            layer_output[:, :, i] += bias[i]
            layer_output[:, :, i][layer_output[:, :, i] < 0] = 0 # Relu
        return layer_output
    
    def pool_layer(layer_input, size=3):
        layer_output = np.zeros((layer_input.shape[0]//size, layer_input.shape[1]//size, layer_input.shape[2]), dtype=np.float32)
        indices = np.zeros((layer_input.shape[0]//size, layer_input.shape[1]//size, layer_input.shape[2], 2), dtype=np.int32)
        for i in range(layer_output.shape[2]):
            for j in range(layer_output.shape[0]):
                for k in range(layer_output.shape[1]):
                    slice_of_input = layer_input[j*size:j*size+size, k*size:k*size+size, i]
                    max_index = np.where(slice_of_input == slice_of_input.max())
                    indices[j, k, i] = np.array([max_index[0][0], max_index[1][0]]) + np.array([j*size, k*size])
                    layer_output[j, k ,i] = slice_of_input.max()
        return layer_output, indices
    
    def flatten(layer_input):
        layer_output  = layer_input.flatten()
        layer_output  = layer_output.reshape((1, layer_output.shape[0]))
        return layer_output
    
    def dense_layer(layer_input, weight, bias, activation='relu'):
        layer_output = layer_input.dot(weight) + bias
        if activation is 'relu':
            layer_output[layer_output < 0] = 0
        elif activation is 'sigmoid':
            layer_output = 1/(1+np.exp(-layer_output))
        elif activation is 'softmax':
            z_exp = np.exp(layer_output)
            layer_output = z_exp / z_exp.sum()
        return layer_output

    conv1                 = conv_layer(frame, weights[0], weights[1])
    pool1, pool1_indices  = pool_layer(conv1)
    conv2                 = conv_layer(pool1, weights[2], weights[3])
    pool2, pool2_indices  = pool_layer(conv2, 2)
#    flat   = flatten(pool2)
#    dense1 = dense_layer(flat,   weights[4],  weights[5])
#    dense2 = dense_layer(dense1, weights[6],  weights[7])
#    dense3 = dense_layer(dense2, weights[8],  weights[9])
#    out    = dense_layer(dense3, weights[10], weights[11], 'softmax')
#    
#    keras_out = model.predict(frame.reshape((1,128,128,3)))
    
    w = np.copy(weights[2][:,:,0,0])
    weights[2][:, :, :, :] = 0
    weights[2][:, :, 0, 0] = w
                
    def unpool(layer_input, indices, shape):
        layer_output = np.zeros(shape, dtype=np.float32)
        for i in range(layer_input.shape[2]):
            for j in range(layer_input.shape[0]):
                for k in range(layer_input.shape[1]):
                    print(indices[j, k, i])
                    layer_output[:,:,i][indices[j, k, i]] = layer_input[j, k, i]
        return layer_output
    
    def deconv(layer_input, weight, bias):
        layer_input[:, :, :][layer_input[:, :, :] < 0] = 0 # Relu
        layer_output = np.zeros(shape, dtype=np.float32)
        for i in range(layer_input.shape[2]):
            for j in range(layer_input.shape[0]):
                for k in range(layer_input.shape[1]):
                    print(indices[j, k, i])
                    layer_output[:,:,i][indices[j, k, i]] = layer_input[j, k, i]
        return layer_output
    
    unpool1 = unpool(pool2, pool2_indices, conv2.shape)
        
    
    
    
    
    
    