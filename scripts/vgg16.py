import os

import h5py
import keras as K
from keras.applications.vgg16 import VGG16


def create_weights_file(name: str, keras_model: K.models.Model, overwrite=True, compression_level=0):
    """
    given a keras model with parameters, this method will create an h5 file
    in the resources with all the weights saved.
    :param name: is the file name, it is not necessary to specify the path or the file extension
    :param keras_model: is the model to be saved
    :param overwrite: is true if u want to overwrite the file in case it already exists
    :param compression_level: an int 0 <= cl <= 9 is the level of compression with the algorithm gzip.
                              it will take more time to create and read the file if the cl is higher but
                              hopefully it will save memory
    :return: NoneObject
    """
    path = '../resources/' + name + '.h5'

    if os.path.isfile(path) and not overwrite:
        return

    with h5py.File(name=path, mode='a') as file:
        main_group = file.create_group(name="root")

        for layer in keras_model.layers:
            group = main_group.create_group(name=layer.name)
            layer_weights = layer.get_weights()
            weights = True
            for l in layer_weights:
                data_set_name = 'weights' if weights else 'biases'
                if 0 < compression_level <= 9:
                    group.create_dataset(name=data_set_name, data=l,
                                         compression="gzip",
                                         compression_opts=compression_level)
                else:
                    group.create_dataset(name='weights' if weights else 'biases', data=l)
                weights = False
            print('Added layer ' + layer.name)


def create_model_file(name: str, keras_model: K.models.Model, overwrite=True):
    """
        given a keras model with parameters, this method will create a txt file
        in the resources with all the layers names in order.
        :param name: is the file name, it is not necessary to specify the path or the file extension
        :param keras_model: is the model to be saved
        :param overwrite: is true if u want to overwrite the file in case it already exists
        :return: NoneObject
        """
    path = '../resources/' + name + '.txt'

    if os.path.isfile(path) and not overwrite:
        return

    with open(path, 'w') as file:
        for layer in keras_model.layers:
            file.write(layer.name)
            if 'pool' in layer.name:
                file.write('.' + str(layer.pool_size[0]) + 'x' + str(layer.pool_size[1]))
            file.write('\n')


def read_weight_file(name):
    path = '../resources/' + name + '.h5'
    with h5py.File(name=path, mode='a') as file:
        file.visit(lambda x: print(x))


if __name__ == '__main__':
    model = VGG16()
    # create_weights_file(name='vgg', keras_model=model)
    # read_weight_file('vgg')
    create_model_file('vgg', keras_model=model)
