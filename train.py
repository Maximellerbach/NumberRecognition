import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from glob import glob
import cv2
import numpy as np

batch_size = 128
num_classes = 12
epochs = 16

# input image dimensions
img_rows, img_cols = 28, 28

x_train=[]
y_train=[]

dos = glob('yourPATH\\NumberRecognition\\label\\*')
for img_path in dos:
    img = cv2.imread(img_path,0)
    img = cv2.resize(img,(img_rows,img_cols))
    img = np.reshape(img,(img_rows,img_cols,1))
    x_train.append(img)
    y_train.append((img_path.split('\\')[-1]).split('_')[0])

input_shape = (img_rows, img_cols, 1)

x_train = np.array(x_train).astype('float32')

x_train /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)

model = Sequential()

model.add(Conv2D(256, (4, 4), activation='relu', input_shape=input_shape))
model.add(Dense(256, activation='relu'))
MaxPooling2D(pool_size=(8, 8), padding='same')

model.add(Conv2D(256, (2, 2), activation='relu', input_shape=input_shape))
model.add(Dense(256, activation='relu'))
MaxPooling2D(pool_size=(4, 4), padding='same')

model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_train, y_train))

model.save('C:\\Users\\maxim\\Desktop\\autre IA\\AI.h5')
