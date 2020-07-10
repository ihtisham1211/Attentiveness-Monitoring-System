import tensorflow as tf
import numpy as np
import pandas as pd
from random import shuffle
import os
import cv2
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Activation, Convolution2D, Dropout, Conv2D
from tensorflow.python.keras.layers import AveragePooling2D, BatchNormalization
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import Input, MaxPooling2D, SeparableConv2D
from tensorflow.python.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.regularizers import l2



# parameters
batch_size = 32
num_epochs = 10000
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = 'training_output/'

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

class DataManager(object):
    def __init__(self, dataset_name='fer2013', dataset_path=None, image_size=(48, 48)):

        self.dataset_name = dataset_name
        self.image_size = image_size
        self.dataset_path = 'datasets/fer2013/fer2013.csv'

    def get_data(self):
        data = self._load_fer2013()
        return data
    
    def _load_fer2013(self):
            data = pd.read_csv(self.dataset_path)
            pixels = data['pixels'].tolist()
            width, height = 48, 48
            faces = []
            for pixel_sequence in pixels:
                face = [int(pixel) for pixel in pixel_sequence.split(' ')]
                face = np.asarray(face).reshape(width, height)
                face = cv2.resize(face.astype('uint8'), self.image_size)
                faces.append(face.astype('float32'))
            faces = np.asarray(faces)
            faces = np.expand_dims(faces, -1)
            emotions = pd.get_dummies(data['emotion']).as_matrix()
            return faces, emotions
                    
def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

def preprocess_input(x):
    x = x.astype('float32')
    x = x / 255.0
         #- data_format mode=tf: will scale pixels between -1 and 1, sample-wise.
    x = x - 0.5
    x = x * 2.0
    return x

def miniXception(input_shape, num_classes):

    #  Base Modulr
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=l2(0.01), use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=l2(0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual Module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=l2(0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=l2(0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = layers.add([x, residual])

    # Residual Module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = layers.add([x, residual])

    # Residual Module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = layers.add([x, residual])

    # Residual Module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = layers.add([x, residual])

    #Output Module
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input, output)
    return model


# model parameters/compilation
model = miniXception(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callbacks
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
    trained_models_path = base_path + dataset_name + '_myModel'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
    tbCallback= TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=False)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr, tbCallback]

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    train_data, val_data = split_data(faces, emotions, validation_split)
    train_faces, train_emotions = train_data
    model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=val_data)



