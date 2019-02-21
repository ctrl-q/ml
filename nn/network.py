import os

import numpy as np

import functions as fct


class NN:

    def __init__(self, hidden_dims=(1024, 2048), datapath=None, model_path=None):
        """Neural network

        Args:
            hidden_dims (tuple): number of hidden units per layer
            datapath (str, path-like): path to the dataset
            model_path (str, path-like): path to the model
        """
        self.hidden_dims = hidden_dims
        # Define functions
        self.activation = fct.relu
        self.softmax = fct.softmax

        self.datapath = datapath
        if datapath:
            with np.load(datapath) as data:
                self.X_train, self.y_train = data["X_train"], data["y_train"]
                self.X_valid, self.y_valid = data["X_valid"], data["y_valid"]
                self.X_test, self.y_test = data["X_test"], data["y_test"]

        self.model_path = model_path
        if model_path and os.path.isfile(model_path):
            with np.load(model_path) as params:
                self.b1 = params["b1"]
                self.W1 = params["W1"]
                self.b2 = params["b2"]
                self.W2 = params["W2"]
                self.b3 = params["b3"]
                self.W3 = params["W3"]

    def predict(self, input_):
        """Predicts the label of x

        Args:
            input_ (array-like): network input

        Returns:
            prediction (int): predicted label
        """
        return self.forward(input_)[-1].argmax(axis=1)

    def initialize_weights(self, method):
        """Initializes layer weights using the chosen method

        Args:
            method (str): one of (Zero, Normal, Glorot)
        """
        method = method.title()
        assert method in ("Zero", "Normal",
                          "Glorot"), "Unsupported initalization method"
        h1, h2 = self.hidden_dims
        b_shapes = [h1, h2, 10]
        self.b1, self.b2, self.b3 = [np.zeros(s) for s in b_shapes]

        W_shapes = [(h1, 784), (h2, h1), (10, h2)]
        if method == "Zero":
            self.W1, self.W2, self.W3 = [np.zeros(s) for s in W_shapes]
        elif method == "Normal":
            self.W1, self.W2, self.W3 = [
                np.random.normal(size=s) for s in W_shapes]
        elif method == "Glorot":
            bounds = [np.sqrt(6 / sum(s)) for s in W_shapes]
            self.W1, self.W2, self.W3 = [
                np.random.uniform(-b, b, s) for b, s in zip(bounds, W_shapes)]

    def forward(self, input_):
        """Propagates input_ through the network

        Args:
            input_ (array-like): network input

        Returns:
            cache (tuple): values of the affine transformations and nonlinearities for all layers
        """

        a1 = np.dot(input_, self.W1.T) + self.b1
        h1 = self.activation(a1)
        a2 = np.dot(h1, self.W2.T) + self.b2
        h2 = self.activation(a2)
        a3 = np.dot(h2, self.W3.T) + self.b3
        h3 = self.softmax(a3, axis=1)
        return input_, a1, h1, a2, h2, a3, h3

    def loss(self, cache, label):
        """Computes the loss between the label and the network's output

        Args:
            cache (sequence): output of self.forward()
            label (int): true label

        Returns:
            cross_entropy: cross-entropy loss between the label and the output
        """
        return fct.crossentropy(label, cache[-1]).mean(axis=0)

    def backward(self, cache, labels, lambda_):
        """Backpropagates through the network

        Args:
            cache (sequence): output of self.forward()
            labels (array-like): labels
            lambda_ (float): weight decay

        Returns:
            gradients (tuple): gradients w.r.t parameters
        """
        input_, a1, h1, a2, h2, a3, h3 = cache
        n = len(labels)
        # gradients w.r.t to nonlinearities and affines
        dl_da3 = h3 - labels

        dl_dh2 = np.dot(dl_da3, self.W3)
        dl_da2 = (a2 > 0) * dl_dh2

        dl_dh1 = np.dot(dl_da2, self.W2)
        dl_da1 = (a1 > 0) * dl_dh1

        # gradients w.r.t parameters
        dl_dW3 = np.dot(dl_da3.T, h2) / n + lambda_ * self.W3
        dl_db3 = dl_da3.mean(axis=0)

        dl_dW2 = np.dot(dl_da2.T, h1) / n + lambda_ * self.W2
        dl_db2 = dl_da2.mean(axis=0)

        dl_dW1 = np.dot(dl_da1.T, input_) / n + lambda_ * self.W1
        dl_db1 = dl_da1.mean(axis=0)
        return dl_db1, dl_dW1, dl_db2, dl_dW2, dl_db3, dl_dW3

    def update(self, grads, eta):
        """Updates MLP parameters based on gradients

        Args:
            grads (sequence): gradients w.r.t parameters
            eta (float): learning rate
        """
        dl_db1, dl_dW1, dl_db2, dl_dW2, dl_db3, dl_dW3 = grads

        self.b1 -= eta * dl_db1
        self.W1 -= eta * dl_dW1
        self.b2 -= eta * dl_db2
        self.W2 -= eta * dl_dW2
        self.b3 -= eta * dl_db3
        self.W3 -= eta * dl_dW3

    def train(self, inputs, labels, batch_size=1, eta=0.1, lambda_=0.0):
        """Trains the MLP for one epoch

        Args:
            inputs (array-like): training inputs
            labels (array-like): training labels
            batch_size (int): batch size
            eta (float): larning rate
            lambda_ (float): weight decay

        Returns:
            loss (float): average loss
        """
        loss = 0.0
        n_batches = inputs.shape[0] // batch_size
        try:
            for i in range(n_batches):
                inputs_i = inputs[i * batch_size:(i + 1) * batch_size]
                labels_i = labels[i * batch_size:(i + 1) * batch_size]

                cache = self.forward(inputs_i)
                grads = self.backward(cache, labels_i, lambda_)
                self.update(grads, eta)

                loss += self.loss(cache, labels_i)
            return loss / n_batches

        finally:
            if self.model_path:
                np.savez_compressed(self.model_path, b1=self.b1,
                                    W1=self.W1, b2=self.b2, W2=self.W2, b3=self.b3, W3=self.W3)

    def test(self, inputs, labels):
        """Calculates loss on a test set

        Args:
            inputs (array-like): training inputs
            labels (array-like): training labels

        Returns:
            loss (float): Model loss
        """
        return self.loss(self.forward(inputs, labels), labels)
