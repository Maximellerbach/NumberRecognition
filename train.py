import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split

from glob import glob
import cv2
import numpy as np

batch_size = 128
num_classes = 12
epochs = 10

# image size and input_size
width, height = 28, 28
input_shape = (28, 28, 1)

x_train=[]
y_train=[]

dos = glob('label\\*')

for img_path in dos:
    #load image in grayscale
    img = cv2.imread(img_path,0)
    img = cv2.resize(img,(width,height))
    img = np.reshape(img,(input_shape))

    x_train.append(img)
    y_train.append((img_path.split('\\')[-1]).split('_')[0])

x_train = np.array(x_train).astype('float32')
x_train /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)


model = Sequential()

model.add(Conv2D(256, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))

model.add(Conv2D(256, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(2,2), padding='same'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

model.save('AI.h5')
