import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool




# Define the ResNet model
class ResNet():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Input layer
        self.X_input = T.tensor4('X_input', dtype=theano.config.floatX)

        # Initial Convolutional Layer
        self.W_init = theano.shared(np.random.randn(3, 3, 3, 64).astype(theano.config.floatX))
        self.b_init = theano.shared(np.zeros(64).astype(theano.config.floatX))
        self.X = conv2d(self.X_input, self.W_init) + self.b_init
        self.X = T.nnet.relu(self.X)

        # Residual Blocks
        self.X = self.identity_block(self.X, 64)
        
        # Pooling Layer
        self.X = pool.pool_2d(self.X, ws=(2, 2), ignore_border=True, mode='average_exc_pad')

        # Flatten
        self.X = T.flatten(self.X, outdim=2)

        # Fully Connected Layer
        self.W_output = theano.shared(np.random.randn(4096, self.num_classes).astype(theano.config.floatX))  # 4096 = 64 * 8 * 8
        self.b_output = theano.shared(np.zeros(self.num_classes).astype(theano.config.floatX))
        self.output_layer = T.nnet.softmax(T.dot(self.X, self.W_output) + self.b_output)

    def identity_block(self, X_input, filters):
        # First convolutional layer
        W1 = theano.shared(np.random.randn(3, 3, filters, filters).astype(theano.config.floatX))
        b1 = theano.shared(np.zeros(filters).astype(theano.config.floatX))
        X = conv2d(X_input, W1) + b1
        X = T.nnet.relu(X)

        # Second convolutional layer
        W2 = theano.shared(np.random.randn(3, 3, filters, filters).astype(theano.config.floatX))
        b2 = theano.shared(np.zeros(filters).astype(theano.config.floatX))
        X = conv2d(X, W2) + b2

        # Adding the shortcut to the output
        X = X_input + X
        X = T.nnet.relu(X)

        return X


    def compile(self):
        Y = T.ivector('Y')
        loss = T.nnet.categorical_crossentropy(self.output_layer, Y).mean()

        # Gradient Descent Parameters
        learning_rate = 0.01
        params = [self.W_init, self.b_init, self.W_res1, self.b_res1, self.W_res2, self.b_res2, self.W_output, self.b_output]
        grads = T.grad(loss, params)

        # Update Rules and Compilation
        updates = [(param, param - learning_rate * grad) for param, grad in zip(params, grads)]
        train_fn = theano.function(inputs=[self.X_input, Y], outputs=loss, updates=updates)

        return train_fn

    def compile_eval(self):
        Y = T.ivector('Y')
        accuracy = T.mean(T.eq(T.argmax(self.output_layer, axis=1), Y))
        eval_fn = theano.function(inputs=[self.X_input, Y], outputs=accuracy)

        return eval_fn

    def compile_roc(self):
        roc_fn = theano.function(inputs=[self.X_input], outputs=self.output_layer)

        return roc_fn