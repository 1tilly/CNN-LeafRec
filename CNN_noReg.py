
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool


class CNN():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Input layer
        self.X_input = T.tensor4('X_input', dtype=theano.config.floatX)
        
        # Convolutional Layer 1
        self.W1 = theano.shared(np.random.randn(3, 3, 3, 16).astype(theano.config.floatX))
        self.b1 = theano.shared(np.zeros(16).astype(theano.config.floatX))
        self.X = conv2d(self.X_input, self.W1) + self.b1
        self.X = T.nnet.relu(self.X)
        
        # Pooling Layer 1
        self.X = pool.pool_2d(self.X, ws=(2, 2), ignore_border=True)
        
        # Convolutional Layer 2
        self.W2 = theano.shared(np.random.randn(3, 3, 16, 32).astype(theano.config.floatX))
        self.b2 = theano.shared(np.zeros(32).astype(theano.config.floatX))
        self.X = conv2d(self.X, self.W2) + self.b2
        self.X = T.nnet.relu(self.X)
        
        # Pooling Layer 2
        self.X = pool.pool_2d(self.X, ws=(2, 2), ignore_border=True)
        
        # Flatten
        self.X = T.flatten(self.X, outdim=2)
        
        # Fully Connected Layer
        self.W_output = theano.shared(np.random.randn(800, self.num_classes).astype(theano.config.floatX))
        self.b_output = theano.shared(np.zeros(self.num_classes).astype(theano.config.floatX))
        self.output_layer = T.nnet.softmax(T.dot(self.X, self.W_output) + self.b_output)
        
    def compile(self):
        Y = T.ivector('Y')
        loss = T.nnet.categorical_crossentropy(self.output_layer, Y).mean()
        
        # Gradient Descent Parameters
        learning_rate = 0.01
        params = [self.W1, self.b1, self.W2, self.b2, self.W_output, self.b_output]
        grads = T.grad(loss, params)
        
        # Update Rules and Compilation
        updates = [(param, param - learning_rate * grad) for param, grad in zip(params, grads)]
        train_fn = theano.function(inputs=[self.X_input, Y], outputs=loss, updates=updates)
        
        return train_fn


    def compile_eval(self):
        Y = T.ivector('Y')
        accuracy = T.mean(T.eq(T.argmax(self.output_layer, axis=1), Y))
        eval_fn = theano.function(inputs=[self.input_layer, Y], outputs=accuracy)
        return eval_fn

    def compile_roc(self):
        roc_fn = theano.function(inputs=[self.input_layer], outputs=self.output_layer)
        return roc_fn