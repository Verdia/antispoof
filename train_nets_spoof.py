import os
import cv2
import random
import argparse
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D, Dense, Dropout, BatchNormalization, Flatten, DepthwiseConv2D, Conv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6

def h_swish(x):
    return x * h_sigmoid(x)

def build_model(input_size, cut_at=-1, unfreeze_from=0):
    # load model
    model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))

    # freeze all layer
    for layer in model.layers:
        layer.trainable = False

    # select layer output
    if cut_at==-1:
        x = model.output
    else:
        x = model.layers[cut_at].output

    # MobileNetV3 Output
    # x = Conv2D(filters=960, kernel_size=(1, 1), strides=1, padding="same")(x)
    # x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    # x = h_swish(x)
    # x = AveragePooling2D(pool_size=(7, 7), strides=1)(x)
    # x = Conv2D(filters=1280, kernel_size=(1, 1), strides=1, padding="same")(x)
    # x = h_swish(x)
    # x = Conv2D(filters=2, kernel_size=(1, 1), strides=1, padding="same")(x)
    # logits = Flatten()(x)
    # predictions = tf.nn.softmax(logits)

    ## GDC Classification Head
    # x = DepthwiseConv2D(int(x.shape[1]), depth_multiplier=1, use_bias=False)(x)
    # x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    # x = Dropout(0.3)(x)
    # x = Conv2D(256, 1, use_bias=False, activation=None, kernel_initializer="glorot_normal")(x)
    # x = Flatten()(x)
    # x = Dense(256, activation=None, use_bias=True, kernel_initializer="glorot_normal")(x)
    # x = Dropout(0.3)(x)
    # x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    # predictions = Dense(1, use_bias=False, activation="sigmoid")(x)

    ## GAP Classification Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    # predictions = Dense(2, activation='softmax')(x)

    # instantiate new model
    myModel = Model(inputs=model.input, outputs=predictions)

    # unfreeze selected layer
    for layer in myModel.layers[unfreeze_from:]:
        layer.trainable = True

    return myModel

def load_image(data_dir, size):
    imgs = []
    print("Load image from ", data_dir, "...")
    for filename in tqdm(glob(data_dir + '/*')):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        img = np.asarray(img)
        img = img / 255
        imgs.append(img)
    return np.asarray(imgs)

def load_data(data_dir, img_size, train_size=0.8):
    random.seed(4)
    fake_data_dir = os.path.join(data_dir, 'spoof')
    fake_data = load_image(fake_data_dir, img_size)
    fake_label = np.zeros(len(fake_data))
    random.shuffle(fake_data)

    idx_train = int(train_size * len(fake_data))

    fake_data_train = fake_data[:idx_train]
    fake_label_train = fake_label[:idx_train]
    fake_data_val = fake_data[idx_train:]
    fake_label_val = fake_label[idx_train:]
    
    real_data_dir = os.path.join(data_dir, 'real')
    real_data = load_image(real_data_dir, img_size)
    real_label = np.ones(len(real_data))
    random.shuffle(real_data)

    idx_train = int(train_size * len(real_data))

    real_data_train = real_data[:idx_train]
    real_label_train = real_label[:idx_train]
    real_data_val = real_data[idx_train:]
    real_label_val = real_label[idx_train:]

    data_train = np.concatenate((real_data_train, fake_data_train))
    label_train = np.concatenate((real_label_train, fake_label_train))
    # one_hot_label_train = to_categorical(label_train)

    if train_size < 1:
        data_val = np.concatenate((real_data_val, fake_data_val))
        label_val = np.concatenate((real_label_val, fake_label_val))
        return data_train, label_train, data_val, label_val
        # one_hot_label_val = to_categorical(label_val)
        # return data_train, one_hot_label_train, data_val, one_hot_label_val

    return data_train, label_train
    # return data_train, one_hot_label_train

def main():
    parser = argparse.ArgumentParser(description = 'Parser to train Face-Anti-Spoofing model')

    parser.add_argument('--data_dir', type=str, required = True, help = 'Data train path consisting of real and fake directory')
    parser.add_argument('--test_dir', type=str, help = 'Data test path consisting of real and fake directory', default=None)
    parser.add_argument('--epoch', type=int, help='Number of epoch', default = 100)
    parser.add_argument('--batch_size', type=int, help='Batch size', default = 32)
    parser.add_argument('--lr', type=float, help='Learning rate', default = 1e-4)
    parser.add_argument('--img_size', type=int, help='Image size', default=224)
    parser.add_argument('--pretrained', type=str, help='Path to pretrained models', default=None)
    parser.add_argument('--cuda', type = str, help = 'CUDA Device ID', default='1')
    parser.add_argument('--save_path', type = str,required=True, help = 'filename to save')
    parser.add_argument('--model_checkpoint', type = str,required=True, help = 'filename to model checkpoint continue training')
    


    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    img_size = (args.img_size, args.img_size)

    model = build_model(img_size, unfreeze_from=-1)

    if args.pretrained:
        model = load_model(args.pretrained, compile=False)
    model.summary()

    data_train, label_train, data_val, label_val = load_data(args.data_dir, img_size)

    if args.test_dir:
        data_test, label_test = load_data(args.test_dir, img_size, 1)

    checkpoint = ModelCheckpoint(args.model_checkpoint, verbose=1, monitor='val_loss',save_best_only=True, mode='auto') 

    optimizer = Adam(args.lr)

    model.compile(
        loss='binary_crossentropy',
        # loss='categorical_crossentropy',
        optimizer=optimizer, 
        metrics=['accuracy'])

    aug = ImageDataGenerator(
        horizontal_flip=True,
        fill_mode="nearest")

    weight0 = len(label_train) / len(label_train[label_train == 0]) * (1 / 2)
    weight1 = len(label_train) / len(label_train[label_train == 1]) * (1 / 2)
    class_weight = {0: weight0, 1: weight1}
    model.fit(
        aug.flow(data_train, label_train,  batch_size=args.batch_size),
        validation_data=(data_val, label_val), batch_size=args.batch_size, epochs=args.epoch, 
        callbacks = [checkpoint],
        class_weight=class_weight, verbose=1)
    model.save(args.save_path+"_basic.h5")
    #model.evaluate(data_test, label_test, batch_size=args.batch_size)

if __name__ == '__main__':
    main()