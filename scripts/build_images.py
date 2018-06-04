import os

import h5py
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from tqdm import tqdm


def write_h5(name: str, images, labels, overwrite=True, compression_level=0):
    """
    given a keras model with parameters, this method will create an h5 file
    in the resources with all the weights saved.
    :param name: is the file name, it is not necessary to specify the path or the file extension
    :param images: are the images to write
    :param labels: are the labels to write, they must be mapped 1:1 with the images
    :param overwrite: is true if u want to overwrite the file in case it already exists
    :param compression_level: an int 0 <= cl <= 9 is the level of compression with the algorithm gzip.
                              it will take more time to create and read the file if the cl is higher but
                              hopefully it will save memory
    :return: NoneObject
    """
    path = '../resources/' + name + '.h5'

    if os.path.isfile(path) and not overwrite:
        return

    with h5py.File(name=path, mode='w') as file:
        main_group = file.create_group(name=name)

        if 0 < compression_level <= 9:
            main_group.create_dataset(name='images', data=images,
                                      compression="gzip",
                                      compression_opts=compression_level)
        else:
            main_group.create_dataset(name='images', data=images)

        main_group.create_dataset(name='labels', data=labels)

        print(name + '.h5 created')


def read_h5(name):
    path = '../resources/' + name + '.h5'
    with h5py.File(name=path, mode='a') as file:
        file.visit(lambda x: print(x))


images_path = '../resources/ILSVRC2012/'
images_test_path = '../resources/test_imgs/'
number_of_images = ''
labels_path = '../resources/ILSVRC2012_validation_ground_truth.txt'
samples = 5
labels_from_vgg = True

if __name__ == '__main__':
    images = []
    labels = []
    path = images_test_path

    net = VGG16()
    with open(labels_path, 'r') as file:
        all_labels = [line for line in file]

    i = 0
    img_list = os.listdir(path)
    for img in tqdm(range(samples)):
        tmp = image.load_img(path + img_list[img], target_size=(230, 230))
        x = image.img_to_array(tmp)
        x = preprocess_input(x)
        images.append(x)

        if labels_from_vgg:
            tmp = image.load_img(path + img_list[img], target_size=(224, 224))
            x = image.img_to_array(tmp)
            x = preprocess_input(x)
            labels.append(net.predict(np.expand_dims(x, axis=0))[0])

        else:
            labels.append(int(all_labels[int(img.split('_')[2].split('.')[0]) - 1]))

    images = np.array(images)
    labels = np.array(labels)
    print('Writing images with shape {}...'.format(np.shape(images)))

    write_h5('images', images, labels, compression_level=9)
    read_h5('images')
