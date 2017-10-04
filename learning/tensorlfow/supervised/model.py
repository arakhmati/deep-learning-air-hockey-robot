import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import regularizers

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

def conv_model(l1=0.00000, l2=0.0001):
    model = Sequential([
                Conv2D(40, 12, activation='relu', name='conv1', kernel_regularizer=regularizers.l2(l2),
                       input_shape=(128, 128, 3)),
                Dropout(0.2, name='dropout1'),
                MaxPooling2D(2, name='pool1'),
                Conv2D(30, 7, activation='relu', name='conv2', kernel_regularizer=regularizers.l2(l2)),
                Dropout(0.2, name='dropout2'),
                MaxPooling2D(2, name='pool2'),
                Conv2D(20, 5, activation='relu', name='conv3', kernel_regularizer=regularizers.l2(l2)),
                Dropout(0.2, name='dropout3'),
                MaxPooling2D(2, name='pool3'),
                Flatten(name='flatten'),
                Dense(500, activation='relu',    name='dense1', kernel_regularizer=regularizers.l2(l2)),
                Dropout(0.2, name='dropout4'),
                Dense(500, activation='relu',    name='dense2', kernel_regularizer=regularizers.l2(l2)),
                Dropout(0.2, name='dropout5'),
                Dense(9,   activation='softmax', name='out', kernel_regularizer=regularizers.l2(l2))
                ])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', 
                  metrics=['accuracy', fmeasure, recall, precision])
    return model

def convlstm_model(l1=0.00000, l2=0.00000):
    model = Sequential([
                ConvLSTM2D(8, 12, activation='relu', name='conv1', input_shape=(None, 128, 128, 3)),
                MaxPooling2D(2, name='pool1'),
                Conv2D(24, 7, activation='relu', name='conv2'),
                MaxPooling2D(2, name='pool2'),
                Conv2D(32, 5, activation='relu', name='conv3'),
                MaxPooling2D(2, name='pool3'),
                Flatten(name='flatten'),
                Dense(512, activation='relu',    name='dense1'),
                Dense(256, activation='relu',    name='dense2'),
                Dense(128, activation='relu',    name='dense3'),
                Dense(64,  activation='relu',    name='dense4'),
                Dense(9,   activation='softmax', name='out')
                ])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', 
                  metrics=['accuracy', fmeasure, recall, precision])
    return model