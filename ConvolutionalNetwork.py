import os
import numpy as np
import theano
import theano.tensor as T
from PIL import Image

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    #data_dir, data_file = os.path.split(dataset)
    # if data_dir == "" and not os.path.isfile(dataset):
    #     # Check if dataset is in the data directory.
    #     new_path = os.path.join(
    #         os.path.split(__file__)[0],
    #         "data",
    #         dataset
    #     )
    # if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
    #     dataset = new_path

    # if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
    #     from six.moves import urllib
    #     origin = (
    #         'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    #     )
    #     print('Downloading data from %s' % origin)
    #     urllib.request.urlretrieve(origin, dataset)

    print('... loading data')
    theano.config.exception_verbosity = 'high'
    #downsampling the images to moderate size (72*96)
    width=720
    height= 960
    new_width, new_height = 560,560
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    # Load the dataset from data folder:
    train_set, valid_set, test_set = ([],[]), ([],[]), ([],[])
    class_map = {}
    class_id, max_class_id = 0,0
    for dir in os.listdir(dataset):
        i = 0
        if dir in class_map:
            class_id = class_map[dir]
        else:
            class_map[dir] = max_class_id
            class_id = max_class_id
            max_class_id += 1
        for b_li in os.walk(dataset+"/"+dir):
            for img_path in b_li[2]:
                # normal size distribution: 50,25,25 for train, valid, test
                with Image.open(dataset+"/"+dir+"/"+img_path) as img:
                    f = np.array([p for p in list(img.crop((left,top,right,bottom)).resize((new_width/20,new_height/20),Image.ANTIALIAS).getdata())], dtype='float64')
                    f /= 255.0
                    if i % 4 == 0:
                        valid_set[0].append(f)
                        valid_set[1].append(class_id)
                    elif i % 4 == 1:
                        test_set[0].append(f)
                        test_set[1].append(class_id)
                    else:
                        train_set[0].append(f)
                        train_set[1].append(class_id)
                i += 1
    train_set = (np.vstack(train_set[0]), np.array(train_set[1]))
    test_set = (np.vstack(test_set[0]), np.array(test_set[1]))
    valid_set = (np.vstack(valid_set[0]), np.array(valid_set[1]))
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets us get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
