import numpy as np


class GradientDescentLinearRegression:
    """
    Linear Regression with gradient-based optimization.
    Parameters
    ----------
    learning_rate : float
        Learning rate for the gradient descent algorithm.
    max_iterations : int
        Maximum number of iteration for the gradient descent algorithm.
    eps : float
        Tolerance level for the Euclidean norm between model parameters in two
        consequitive iterations. The algorithm is stopped when the norm becomes
        less than the tolerance level.
    """

    def __init__(self, learning_rate=1, max_iterations=10000, eps=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.eps = eps

    def predict(self, X):
        """Returns predictions array of shape [n_samples,1]"""
        return np.dot(X, self.w.T)

    def cost(self, X, y):
        """Returns the value of the cost function as a scalar real number"""
        y_pred = self.predict(X)
        loss = (y - y_pred) ** 2
        return np.mean(loss)

    def grad(self, X, y):
        """Returns the gradient vector"""
        y_pred = self.predict(X)
        d_intercept = -2 * sum(y - y_pred)  # dJ/d w_0.
        d_x = -2 * sum(X[:, 1:] * (y - y_pred).reshape(-1, 1))  # dJ/d w_i.
        g = np.append(np.array(d_intercept), d_x)  # Gradient.
        return g / X.shape[0]  # Average over training samples.

    def adagrad(self, g):
        self.G += g ** 2  # Update cache.
        step = self.learning_rate / (np.sqrt(self.G + self.eps)) * g
        return step

    def fit(self, X, y, method="adagrad", verbose=True):
        """
        Fit linear model with gradient descent.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_predictors]
            Training data
        y : numpy array of shape [n_samples,1]
            Target values.
        method : string
                 Defines the variant of gradient descent to use.
                 Possible values: "standard", "adagrad".
        verbose: boolean
                 If True, print the gradient, parameters and the cost function
                 for each iteration.

        Returns
        -------
        self : returns an instance of self.
        """

        self.w = np.zeros(X.shape[1])  # Initialization of params.
        if method == "adagrad":
            self.G = np.zeros(X.shape[1])  # Initialization of cache for AdaGrad.
        w_hist = [self.w]  # History of params.
        cost_hist = [self.cost(X, y)]  # History of cost.

        for iter in range(self.max_iterations):

            g = self.grad(X, y)  # Calculate the gradient.
            if method == "standard":
                step = self.learning_rate * g  # Calculate standard gradient step.
            elif method == "adagrad":
                step = self.adagrad(g)  # Calculate AdaGrad step.
            else:
                raise ValueError("Method not supported.")
            self.w = self.w - step  # Update parameters.
            w_hist.append(self.w)  # Save to history.

            J = self.cost(X, y)  # Calculate the cost.
            cost_hist.append(J)  # Save to history.

            if verbose:
                print(f"Iter: {iter}, gradient: {g}, params: {self.w}, cost: {J}")

            # Stop if update is small enough.
            if np.linalg.norm(w_hist[-1] - w_hist[-2]) < self.eps:
                break

        # Final updates before finishing.
        self.iterations = iter + 1  # Due to zero-based indexing.
        self.w_hist = w_hist
        self.cost_hist = cost_hist
        self.method = method

        return self

    
