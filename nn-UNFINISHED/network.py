import numpy as np
import functions as fct
from copy import deepcopy
from matplotlib import pyplot as plt


class Network:

    def __init__(self, layer_sizes, hidden_activ_fct=fct.rect, output_activ_fct=fct.softmax, lambd=([0, 0], [0, 0])):
        """
        Args:
            layer_sizes (list): Number of neurons per layer (including input)
            method (str): Either "loop" or "matrix". Defines how bprop should be calculated
        """
        self.depth = len(layer_sizes)
        self.layers = []
        self.lambd = lambd

        for i in range(1, self.depth):
            n = layer_sizes[i]
            n_c = layer_sizes[i-1]
            activ_fct = output_activ_fct if i == self.depth - 1 else hidden_activ_fct
            self.layers.append(self.Layer((n, n_c), activ_fct))

    def _fprop(self, X):
        activation = X.transpose()
        self.activations = [activation]
        for layer in self.layers:
            activation = layer._fprop(activation)
            self.activations.append(activation)

    def _bprop(self, X, Y):
        n = 1 if isinstance(Y, (int, np.intp)) else len(Y)  # number of examples
        self._fprop(X)
        os = self.activations[-1]
        self.grad_oa = os / n
        self.grad_oa[int(Y)] -= 1

        hs = self.activations[-2]
        w2 = self.layers[-1].weights
        w1 = self.layers[-2].weights

        self.grad_w2 = (np.outer(self.grad_oa, hs) + self.lambd[1][0] * np.sign(w2) + 2 * self.lambd[1][1] * w2) / n
        self.grad_b2 = self.grad_oa

        self.grad_hs = np.dot(np.transpose(w2), self.grad_oa)
        self.grad_ha = self.grad_hs * \
            np.where(self.layers[-2].preactivation(X) > 0, 1, 0)
        self.grad_w1 = (np.outer(self.grad_ha, X) + self.lambd[0][0] * np.sign(w1) + 2 * self.lambd[0][1] * w1) / n
        self.grad_b1 = self.grad_ha

    def f(self, X):
        return self.activations[-1]

    def train(self, X, Y, max_iter, eta, K, method="matrix", check_gradient=False, compute_losses=False, X_valid=None, y_valid=None, X_test=None, y_test=None):        
        assert method in ("matrix", "loop"), "Method can only be 'matrix' or 'loop'"
        self.max_iter = max_iter
        Y = int(Y) if isinstance(Y, (float, int)) else np.asarray(
            Y, dtype="intp")  # Floats are not accepted as array indices

        n = 1 if isinstance(Y, int) else len(Y)  # number of examples
        if n > 1:
            batch_indexes = np.random.permutation(np.arange(n))
            X_batches = (
                X[batch_indexes[K*i:K*(i+1)],:]
                for i in range(n//K)
            )
            Y_batches = (
                Y[batch_indexes[K*i:K*(i+1)]]
                for i in range(n//K)
            )

            batches = zip(X_batches, Y_batches)

        else:
            batches = [[X, Y]]

        if compute_losses:
            errors = np.empty((max_iter, 3))
            losses = np.empty((max_iter, 3))
            

        if method == "matrix":
            for i in range(max_iter):
                for X_batch, Y_batch in batches:
                    self._bprop(X_batch, Y_batch)
                    self.layers[0].weights -= eta*self.grad_w1
                    self.layers[0].biases -= eta*self.grad_b1
                    self.layers[1].weights -= eta*self.grad_w2
                    self.layers[1].biases -= eta*self.grad_b2
            if compute_losses:
                errors[i,:] = self.error(X, Y), self.error(X_valid, y_valid), self.error(X_test, y_test)
                losses[i,:] = self.loss(X, Y), self.loss(X_valid, y_valid), self.loss(X_test, y_test)

        elif method == "loop":
            for i in range(max_iter):
                for X_batch, Y_batch in batches:
                    # The next line is the only difference between using a loop or a matrix
                    for x, y in zip(X_batch, Y_batch):
                        self._bprop(x, y)
                        self.layers[0].weights -= eta*self.grad_w1
                        self.layers[0].biases -= eta*self.grad_b1
                        self.layers[1].weights -= eta*self.grad_w2
                        self.layers[1].biases -= eta*self.grad_b2

            if compute_losses:
                errors[i,:] = self.error(X, Y), self.error(X_valid, y_valid), self.error(X_test, y_test)
                losses[i,:] = self.loss(X, Y), self.loss(X_valid, y_valid), self.loss(X_test, y_test)


        if check_gradient:
            epsilon = np.random.uniform(10**(-6), 10**(-4))
            assert self._finite_difference_check(X, Y, epsilon)

        if compute_losses:
            np.savetxt("errors.txt", errors, header="Train, valid, test")
            np.savetxt("losses.txt", losses, header="Train, valid, test")

    def predict(self, X):
        if X.ndim == 1:
            return np.argmax(self.f(X)[-1])
        else:
            return [np.argmax(self.f(x)[-1]) for x in X]

    def plot_decision_regions(self, X_train, y_train, X_test, Y_test):
            # Decision boundary
            predictions_train = self.predict(X_train)
            plt.scatter(X_train[:,0],X_train[:,1],c=predictions_train)
            predictions_test = self.predict(X_test)
            plt.scatter(X_test[:,0],X_test[:,1],c=predictions_test)
            plt.title('Prediction. Lambda: {0}, hidden units: {1}, iterations: {2}'.format(self.lambd[0][1], self.layers[0].n, self.max_iter))
            plt.show()           

    def perturb(self, epsilon):
        assert 10**(-6) <= epsilon <= 10**(-4), "epsilon must be between 10^-6 and 10^-4"
        perturbed_net = deepcopy(self)
        for layer in perturbed_net.layers:
            layer.weights, layer.biases = layer.perturb(epsilon)
        return perturbed_net

    def _finite_difference_check(self, X, Y, epsilon):
        old_loss = self.loss(X, Y)

        perturbed_net = self.perturb(epsilon)
        new_loss = perturbed_net.loss(X, Y)

        try:
            return (0.99 <= new_loss/old_loss <= 1.01) or (np.isnan(old_loss/new_loss))
        except ZeroDivisionError:
            return old_loss == new_loss

    def loss(self, X, Y):
        loss = -np.mean(np.log(self.f(X)[Y]))

        return loss
        if X.ndim == 1:
            return -np.log(self.f(X)[Y])
        else:
            return -np.mean(np.log(self.f(X[i]))[Y[i]] for i in range(X.shape[0]))
    
    def error(self, X, Y):
        if X.ndim == 1:
            return 1.0 - float(self.predict((X)) == Y)
        else:
            return 1.0 - np.mean(self.predict(X[i]) == Y[i] for i in range(X.shape[0]))
                

    class Layer:
        def __init__(self, weights_shape, activ_fct):
            self.n, self.n_c = weights_shape
            self.activ_fct = activ_fct
            self.biases = np.zeros((self.n))

            bound = 1.0/np.sqrt(self.n_c)
            self.weights = np.random.uniform(-bound, bound, weights_shape)

        def _fprop(self, x):
            return self.activ_fct(self.preactivation(x))

        def preactivation(self, x):
            return np.dot(self.weights, x) + self.biases

        def perturb(self, epsilon):
            return self.weights + epsilon, self.biases + epsilon
