import os

dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
sys.path.append(dir_path + '/../utils')
sys.path.append(dir_path + '/../tensorflow')

from keras.models import load_model
from model import fmeasure, recall, precision, conv_model

import numpy as np
from caffe2.python import workspace
import mobile_exporter

import keras_to_caffe2

# Load keras model
keras_model = load_model(dir_path + '/../tensorflow/models/model.h5', 
                         {'fmeasure': fmeasure, 'recall': recall, 'precision': precision})
#keras_model = conv_model()
#print(keras_model.layers[0].get_weights())


# Copy from keras to caffe2
caffe2_model = keras_to_caffe2.keras_to_caffe2(keras_model)

# Generate random data
frame = np.random.random_sample((1, 9, 128, 128)).astype(np.float32)

# Test 
keras_pred = keras_model.predict(frame)[0]
workspace.FeedBlob('in', frame)
workspace.RunNet(caffe2_model.net)
caffe2_pred = workspace.FetchBlob('softmax')[0]

for a, b in zip(keras_pred, caffe2_pred):
    print('%e %e' % (a, b))
print('%d %d' % (np.argmax(keras_pred), np.argmax(caffe2_pred)))


## Test on a lot of data
#from data_utils import load_data
#import matplotlib.pyplot as plt
#frames, labels = load_data('../data/2017-11-09-00-22-45_500.h5')
#for i, frame in enumerate(frames):
#    plt.imshow(frame[0:3].transpose((1, 2, 0)))
#    plt.show()
#    frame = frame.reshape(1, 9, 128, 128)
#    print(frame.mean())
#    keras_pred = keras_model.predict(frame)[0]
#    
#    workspace.FeedBlob('in', frame)
#    workspace.RunNet(caffe2_model.net)
#    caffe2_pred = workspace.FetchBlob('softmax')
#    print(keras_pred, caffe2_pred)
#    print('%d %d' % (np.argmax(keras_pred), np.argmax(caffe2_pred)))

# Export caffe2 to Android
init_net, predict_net = mobile_exporter.Export(workspace, caffe2_model.net, caffe2_model.params)
with open('init_net.pb', 'wb') as f:
    f.write(init_net.SerializeToString())
with open('predict_net.pb', 'wb') as f:
    f.write(predict_net.SerializeToString())