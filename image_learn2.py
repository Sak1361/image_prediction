from flickrapi import FlickrAPI
from PIL import Image
from sklearn import model_selection
from urllib.request import urlretrieve
from tensorflow import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.applications import VGG16
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


def generate_data(classes, image_size, npy_name):
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
            data = np.asarray(image)
            X.append(data)
            Y.append(index)
    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y)
    xy = (X_train, X_test, Y_train, Y_test)
    np.save(npy_name, xy)
    print('success!')


def vgg(classes, image_size, npy_name, learn_file):
    # データ読み込み
    num_classes = len(classes)
    X_train, X_test, Y_train, Y_test = np.load(npy_name, allow_pickle=True)
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)
    X_train = X_train.astype('float') / 255.0
    X_test = X_test.astype('float') / 255.0
    # モデルの定義
    model = VGG16(weights='imagenet', include_top=False,
                  input_shape=(image_size, image_size, 3))
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes, activation='softmax'))

    model = Model(inputs=model.input, outputs=top_model(model.output))

    for layer in model.layers[:15]:
        layer.trainable = False

    opt = Adam(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=32, epochs=60)

    score = model.evaluate(X_test, Y_test, batch_size=32)
    print(score)
    model.save(learn_file)


def predict(classes, image_size, learn_file, image_file):
    image = Image.open(image_file)
    image = image.convert('RGB').resize((image_size, image_size))
    data = np.asarray(image) / 255.0
    X = []
    X.append(data)
    X = np.array(X)

    # モデルのロード
    model = load_model(learn_file)
    result = model.predict([X])[0]
    predicted = result.argmax()  # 大きい方
    percentage = int(result[predicted] * 100)

    print(classes[predicted], percentage)


if __name__ == "__main__":
    classes = sys.argv[1:]
    key = '40680b9cb8107306558ebe7a038cfd73'
    secret_key = '8238b81113dac77b'
    npy_file = 'imagefiles_bike.npy'
    learn_file = 'vgg16_bike.h5'
    image_size = 224
    np.load = partial(np.load, allow_pickle=True)  # np.loadエラー回避
    ask = input("saving=1,generate=2,learn=3,predict=4：")
    if ask == '1':
        for c in classes:
            get_image(c, key, secret_key)
            print(c)
    elif ask == '2':
        generate_data(classes, image_size, npy_file)
    elif ask == '3':
        vgg(classes, image_size, npy_file, learn_file)
    elif ask == '4':
        image_file = input('predict imagefile?：')
        predict(classes, image_size, learn_file, image_file)
    else:
        print('にゃーん')
