
import numpy as np
import theano
import theano.tensor as T

# Importing TensorFlow to access the leaf dataset
import tensorflow_datasets as tfds


def preprocess(dataset, info):
    images, labels = [], []
    for image, label in dataset:
        images.append(np.array(image) / 255.0)
        labels.append(np.eye(info.features['label'].num_classes)[label])
    return np.array(images), np.array(labels)

# Updated function for sharing dataset with Theano
def shared_dataset(data_x, data_y, borrow=True):
    """
    Function that loads the dataset into shared variables.
    
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch every time
    is needed (the default behaviour if the data is not in a shared variable)
    would lead to a large decrease in performance.
    """
    
    # Ensure the data is in the correct float format
    data_x = np.asarray(data_x, dtype=theano.config.floatX)
    data_y = np.asarray(data_y, dtype=theano.config.floatX)
    
    # Create Theano shared variables
    shared_x = theano.shared(data_x, borrow=borrow)
    shared_y = theano.shared(data_y, borrow=borrow)
    
    # Cast the labels back to integers for indexing during training
    return shared_x, T.cast(shared_y, 'int32')


def load_and_split_dataset():
    # Load the leaf dataset
    dataset, info = tfds.load('plant_leaves', with_info=True, as_supervised=True, split='train')
    
    # Get the number of classes
    num_classes = info.features['label'].num_classes
    
    # Calculate the total number of samples
    total_samples = info.splits['train'].num_examples
    
    # Define the ratio for train, test, and eval
    train_ratio = 0.7
    test_ratio = 0.2
    eval_ratio = 0.1 # not used, but for the sake of completeness I included it
    
    # Calculate the number of samples for each set
    train_size = int(train_ratio * total_samples)
    test_size = int(test_ratio * total_samples)
    eval_size = total_samples - train_size - test_size # not used, but for the sake of completeness I included it
    
    # Create the dataset splits
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size).take(test_size)
    eval_dataset = dataset.skip(train_size + test_size)
    
    # Preprocess the datasets
    train_images, train_labels = preprocess(train_dataset, info)
    test_images, test_labels = preprocess(test_dataset, info)
    eval_images, eval_labels = preprocess(eval_dataset, info)
    
    # Convert the data into shared variables
    train_set_x, train_set_y = shared_dataset(train_images, train_labels)
    test_set_x, test_set_y = shared_dataset(test_images, test_labels)
    eval_set_x, eval_set_y = shared_dataset(eval_images, eval_labels)
    
    return (train_set_x, train_set_y), (test_set_x, test_set_y), (eval_set_x, eval_set_y), num_classes



if __name__ == '__main__':
    train_set, test_set, eval_set, num_classes = load_and_split_dataset()
    print(f"Number of classes: {num_classes}")
    print(f"Training set: {len(train_set[0].get_value())}")

    # Print shape of training sample
    print(f"Training set shape: {train_set[0].get_value().shape}")
    
    #show example image
    import matplotlib.pyplot as plt
    plt.imshow(train_set[0].get_value()[0])
