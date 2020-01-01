from flickrapi import FlickrAPI
from PIL import Image
from sklearn import model_selection
from urllib.request import urlretrieve
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from functools import partial  # kerasでnp.loadエラー起こすのを防ぐ
import numpy as np
import os
import time
import sys
import glob


def get_image(save_dir, key, secret_key):
    wait = 1
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    flickr = FlickrAPI(key, secret_key, format='parsed-json')
    result = flickr.photos.search(
        text=key,
        per_page=400,
        media='photos',
        sort='relevance',
        safe_search=1,
        extras='url_q,license'
    )

    photos = result['photos']

    for i, photo in enumerate(photos['photo']):
        url_q = photo['url_q']
        file_p = f'{save_dir}/{photo["id"]}.jpg'
        if os.path.exists(file_p):
            continue
        urlretrieve(url_q, file_p)
        # time.sleep(wait)
        print(i)


def generate_data(classes, image_size):
    num_class = len(classes)
    # 画像読み込みとnumpy配列への変換
    X = []
    Y = []
    for index, class_label in enumerate(classes):
        photo_dir = class_label
        files = glob.glob(photo_dir + '/*.jpg')
        for i, file in enumerate(files):
            image = Image.open(file)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image) / 255.0
            X.append(data)
            Y.append(index)
    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y)
    xy = (X_train, X_test, Y_train, Y_test)
    np.save('image_files.npy', xy)


def cnn(classes, image_size):
    num_classes = len(classes)
    X_train, X_test, Y_train, Y_test = np.load('image_files.npy')
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)
    # モデルの定義
    model = Sequential()
    # フィルター32枚で畳み込み
    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=(image_size, image_size, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # pooling
    model.add(Dropout(0.25))
    # フィルター64枚で再度畳み込み
    model.add(Conv2D(64, (3, 3), activation='relu',
                     input_shape=(image_size, image_size, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    #opt = SGD(lr=0.01)
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=32, epochs=20)

    score = model.evaluate(X_test, Y_test, batch_size=32)


if __name__ == "__main__":
    classes = sys.argv[1:]
    key = '40680b9cb8107306558ebe7a038cfd73'
    secret_key = '8238b81113dac77b'
    image_size = 150
    np.load = partial(np.load, allow_pickle=True)  # np.loadエラー回避
    ask = input("saving=1,generate=2,learn=3：")
    if ask == '1':
        for c in classes:
            get_image(c, key, secret_key)
            print(c)
    elif ask == '2':
        generate_data(classes, image_size)
    elif ask == '3':
        cnn(classes, image_size)
    else:
        pass
