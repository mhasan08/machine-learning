import numpy as np
import json
import os
from PIL import Image


def import_data():
    classes = np.array(['non-cat', 'cat'])
    test_data_size = 50;

    test_data_path = 'test_data'
    test_data_annotation_path = 'test_data.json'

    train_data_path = 'train_data'
    train_data_annotation_path = 'train_data.json'

    test_data = []
    test_data_annotation = []

    # test data

    with open(test_data_annotation_path, 'r') as fp:
        test_data_annotation_json = json.load(fp)

    for imageName in os.listdir(test_data_path):
        # print imageName
        annot_name = imageName[:len(imageName) - 4]
        img = Image.open(test_data_path + "/" + imageName)
        arr = np.array(img)
        test_data.append(arr)

        # anotations
        is_cat = test_data_annotation_json[str(annot_name)]
        test_data_annotation.append(int(is_cat))

    test_data_final = np.array(test_data)
    print(test_data_final.shape)
    test_data_annotation_final = np.array(test_data_annotation).reshape(test_data_size, 1).T
    print(test_data_annotation_final.shape)

    # train data
    train_data_size = 243

    train_data = []
    train_data_annotation = []

    with open(train_data_annotation_path, 'r') as fp:
        train_data_annotation_json = json.load(fp)

    for imageName in os.listdir(train_data_path):
        # print imageName
        annot_name = imageName[:len(imageName) - 4]
        img = Image.open(train_data_path + "/" + imageName)
        arr = np.array(img)
        train_data.append(arr)

        # anotations
        is_cat = train_data_annotation_json[str(annot_name)]
        train_data_annotation.append(int(is_cat))

    train_data_final = np.array(train_data)
    print(train_data_final.shape)
    train_data_annotation_final = np.array(train_data_annotation).reshape(train_data_size, 1).T
    print(train_data_annotation_final.shape)

    return train_data_final, train_data_annotation_final, test_data_final, test_data_annotation_final, classes

# import_data()
