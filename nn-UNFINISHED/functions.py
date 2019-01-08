import numpy as np

## Activation functions
rect = lambda x: np.where(x > 0, x, 0)

def softmax(x):
    if x.ndim == 1:
        b = x.max()
        y = np.exp(x - b)
        return y / y.sum()
    else:
        b = x.max(axis=1).reshape(x.shape[0],1)
        y = np.exp(x - b)
        return y / y.sum(axis=1)


## Helper functions

def display_gradients(network, X_train, y_train, K, method="matrix"):
    """Displays the gradients using direct computation & finite difference    """
    # The finite difference check is done during training
    # If the following line completes without error, then the check passed
    network.train(X=X_train,Y=y_train,max_iter=1000,eta=0.01,K=K, method=method, check_gradient=True)
    def print_grads(net):
        grads = {"grad_oa":net.grad_oa,"grad_w2":net.grad_w2,"grad_b2":net.grad_b2,"grad_hs":net.grad_hs,"grad_ha":net.grad_ha,"grad_w1":net.grad_w1,"grad_b1":net.grad_b1}
        for name, value in grads.items():
            print(name, value)
            print()
        return grads

    print("DIRECT COMPUTATION OF GRADIENT")
    print("-"*50)
    print_grads(network)

    perturbed_net = network.perturb(epsilon=np.random.uniform(10**(-6), 10**(-4)))
    perturbed_net.train(X=X_train,Y=y_train,max_iter=10,eta=1,K=K, method=method)
    print("-"*50)
    print("FINITE DIFFERENCE GRADIENT")
    print("-"*50)
    print_grads(perturbed_net)
