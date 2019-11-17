import numpy as np
import os
import tensorflow as tf
import sys
import cv2


def hex_to_ascii(dir_name):
    return bytearray.fromhex(dir_name).decode()


def process_img(img_path):
    size = 64
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))  # rescale image
    return img


def get_label(name):
    return int(name.split("_")[1])


def create_label_dictionary():
    id = 0
    id_dict = dict()
    for code in range(ord('0'), ord('9') + 1):
        id_dict[chr(code)] = id
        id += 1
    for code in range(ord('a'), ord('z') + 1):
        id_dict[chr(code)] = id
        id += 1
    for code in range(ord('A'), ord('Z') + 1):
        id_dict[chr(code)] = id
        id += 1
    print("created label dictionary : {}".format(id_dict))
    return id_dict


def character_to_id(id_dict, char):
    return id_dict[char]


def id_to_character(id_dict, id):
    for key, item in id_dict.items():
        if item is id:
            return key


def save_as_np_arr(filepath, size, id_dict, label):
    print("saving image and label as numpy array")
    images = np.zeros(shape=(size, 64, 64, 1))  # Array with all images
    labels = np.zeros(shape=(size, n_labels))  # Array with all labels, one-hot encoded
    for i, file in enumerate(os.listdir(filepath)):
        img = cv2.imread(os.path.join(filepath, file), flags=cv2.IMREAD_GRAYSCALE)
        for j, x in enumerate(img):
            for k, y in enumerate(x):
                images[i][j][k][0] = y
        cls_id = character_to_id(id_dict, file.split("_")[1])
        labels[i, cls_id] = 1
    np.save("nist_labels_test", labels)
    np.save("nist_images_test", images)
    print("Saved labels and images as numpy array!")
    return images, labels


if __name__ == "__main__":
    n_labels = 62
    source_path = "C:\\Users\\Jane\\Downloads\\by_class\\by_class"  # Path to images
    target_path = "C:\\Users\\Jane\\Desktop\\nist_img_test"
    hsf_0 = "hsf_0"
    img_id = 0
    flag_rescale_img = False
    total_images = 1240

    if flag_rescale_img:
        print("started processing image")
        # rescale all images
        for _, dir in enumerate(os.listdir(source_path)):
            character = hex_to_ascii(dir)
            print("processing image in '{}'".format(character))
            new_path = os.path.join(source_path, dir, hsf_0)
            for i, file in enumerate(os.listdir(new_path)):
                new_img_name = "class_{}_{}.png".format(character, img_id)
                img_id += 1
                path = os.path.join(target_path, new_img_name)
                image = process_img(os.path.join(new_path, file))
                cv2.imwrite(path, image)
                if i is 19:
                    break
        print("image processing completed! total {} are saved into new path".format(img_id))

    id_dict = create_label_dictionary()
    save_as_np_arr(target_path, total_images, id_dict, n_labels)
