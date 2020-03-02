"""
Author : Tham Yik Foong
Student ID : 20200786
Project title : Conditional Variational Autoencoder implementation with Pythonn

Academic integrity statement :
I, Tham Yik Foong, have read and understood the School's Academic Integrity Policy, as well as guidance relating to
this module, and confirm that this submission complies with the policy. The content of this file is my own original
work, with any significant material copied or adapted from other sources clearly indicated and attributed.
"""

import numpy as np
import os
import cv2
import string


def hex_to_ascii(dir_name):
    return bytearray.fromhex(dir_name).decode()


def process_img(img_path, size=64):
    """
    Rescale all images in a path
    :param img_path: source path of images
    :param size: rescaled size
    :return: rescaled images
    """
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))  # rescale image
    return img


def create_label_dictionary():
    """
    Creating a list of 0 - 9, a to z and A to Z
    :return: Dictionary with 62 elements, store id as key, and character as value
    """
    return {s: i for i, s in enumerate(string.digits + string.ascii_letters)}


def character_to_id(id_dict, char):
    """
    Return id of character storing in label dictionary
    :return: id of character
    """
    return id_dict[char]


def id_to_character(id_dict, id):
    """
    Return character based on id storing in label dictionary
    :return: character of the given id
    """
    _id = id.item()
    for key, item in id_dict.items():
        if item is _id:
            return key


def save_as_np_arr(filepath, size, id_dict, label):
    """
    Saving images and labels as two separated numpy array in current directory
    :param filepath: source of images file path
    :param size: number of total images
    :param id_dict: label dictionary
    :param label: number of labels
    :return: array of all images data and one-hot encoded label
    """
    print("saving image and label as numpy array")
    images = np.zeros(shape=(size, 64, 64))  # Array with all images
    labels = np.zeros(shape=(size, label))  # Array with all labels, one-hot encoded
    for i, file in enumerate(os.listdir(filepath)):
        img = cv2.imread(os.path.join(filepath, file), flags=cv2.IMREAD_GRAYSCALE)
        images[i] = np.array(img)  # removed channel to avoid nested loop, since images only required 1 channel
        cls_id = character_to_id(id_dict, file.split("_")[1])
        labels[i, cls_id] = 1
    images = images / 255
    np.save("nist_labels", labels)
    np.save("nist_images", images)
    print("Saved labels and images as numpy array!")
    return images, labels


def main():
    n_labels = 62
    source_path = "C:\\Users\\Jane\\Downloads\\by_class\\by_class"  # Path to images
    target_path = "C:\\Users\\Jane\\Desktop\\nist_img_test"
    hsf_0 = "hsf_0"
    total_img_from_class = 300
    img_id = 0
    flag_rescale_img = False
    total_images = 174415
    rescaled_size = 64

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
                image = process_img(os.path.join(new_path, file), rescaled_size)
                cv2.imwrite(path, image)
                if i is total_img_from_class-1:
                    break
        print("image processing completed! total {} are saved into new path".format(img_id))

    id_dict = create_label_dictionary()
    save_as_np_arr(target_path, total_images, id_dict, n_labels)


if __name__ == "__main__":
    main()
