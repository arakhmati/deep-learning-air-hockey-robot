from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, LSTM
#from keras.models import load_model

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import numpy as np

data = np.load('/home/ahmed/Documents/41X/supervised_data/game_000000_data.npz')
images = data['images']
labels = data['labels']

normalized_images = (images.astype(np.float)-128)/128

normalized_images_train, normalized_images_test, labels_train, labels_test = train_test_split(normalized_images, labels, test_size=0.2)

enc = OneHotEncoder()
enc.fit(labels+1)

one_hot_encoded_labels_train = enc.transform(labels_train+1).toarray()
x_train = one_hot_encoded_labels_train[:, :3]
y_train = one_hot_encoded_labels_train[:, 3:]

one_hot_encoded_labels_test = enc.transform(labels_test+1).toarray()
x_test = one_hot_encoded_labels_test[:, :3]
y_test = one_hot_encoded_labels_test[:, 3:]

del enc, one_hot_encoded_labels_test, one_hot_encoded_labels_train


def build_model():
    image_input = Input(shape=(128, 128, 3), dtype='float32')
    conv1  = Conv2D(4, 10, activation='relu')(image_input)
    pool1  = MaxPooling2D(3)(conv1)
    conv2  = Conv2D(8, 10, activation='relu')(pool1)
    pool2  = MaxPooling2D(3)(conv2)
    flat   = Flatten()(pool2)
    dense1 = Dense(512, activation='relu')(flat)
    dense2 = Dense(256, activation='relu')(dense1)
    
    x0 = Dense(128, activation='relu')(dense2)
    x1 = Dense(128, activation='relu')(x0)
    x2 = Dense(3, activation='softmax')(x1)
    
    
    y0 = Dense(128, activation='relu')(dense2)
    y1 = Dense(128, activation='relu')(y0)
    y2 = Dense(3, activation='softmax')(y1)
  
    model = Model(inputs=[image_input], outputs=[x2, y2])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

model = build_model()
model.fit([normalized_images_train], [x_train, y_train], epochs=20, batch_size=256)

x_pred, y_pred = model.predict([normalized_images_test], batch_size=1024)
x_pred = (x_pred > 0.5).astype(np.float)
y_pred = (y_pred > 0.5).astype(np.float)

print(confusion_matrix(list(map(lambda x: np.argmax(x), x_test)), list(map(lambda x: np.argmax(x), x_pred))))
print(confusion_matrix(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), y_pred))))

model.save('../model.h5')