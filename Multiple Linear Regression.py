import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'Housing_dataset.csv')
Y = df[['price']]
X = df.drop(['price'], axis=1)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X['house_size'], X['room_number'], Y)
plt.show()

# feature Normalization

x = X.values
y = Y.values


# normalizing the data for starting the regression (standardizing)
def featureNormalize(x_m):
    mu = np.zeros((1, x_m.shape[1]))  # mean of each column of each column of x feature matrix
    sigma = np.zeros((1, x_m.shape[1]))  # std deviation of each column of x feature matrix
    # storing all the values of x data frame in a floate type
    # in a new variable to operate on
    x_norm = x_m.astype(float)
    for i in range(0, len(mu) + 1):
        mu[:, i] = x_m[:, i].mean()
        sigma[:, i] = x_m[:, i].std()
        # normal function done on each element in x matrix
        x_norm[:, i] = (x_m[:, i] - mu[:, i]) / sigma[:, i]
    return x_norm, mu, sigma


#
x_norm, mu, sigma = featureNormalize(x)
# adding a column of 1's in the modified  feature matrix to apply the dot product on it with the weight matrix
x_norm = np.concatenate((np.ones(len(x_norm)).reshape(-1, 1), x_norm), axis=1)


def computeCost(x, y, theta):
    n = len(y)
    h_x = np.dot(x, theta)
    j = np.sum(np.square(h_x - y)) / (2 * n)
    return j


def gradientDescentMulti(X, Y, theta, alpha, iterations):
    m = len(Y)
    p = np.copy(X)
    t = np.copy(theta)
    j = []
    print('Running Gradient Descent')
    for i in range(0, iterations + 1):
        cost = computeCost(p, Y, t)
        j.append(cost)
        h_x = np.dot(p, t)
        loss = h_x - Y
        for f in range(theta.size):
            t[f] = t[f] - alpha / m * (np.sum((np.dot(p[:, f].T, loss))))
    return j, t


def plot_j_hist(j_hist, iterations):
    plt.plot(np.arange(len(j_hist)), j_hist)
    plt.xlabel("Number of iterations (Epochs)")
    plt.ylabel("Cost function  J(Θ)")
    plt.title("Gradient Descent")
    plt.show()


def main():
    theta = np.zeros((3, 1))
    iterations = 1500
    alpha = 0.01
    j_hist, theta = gradientDescentMulti(x_norm, Y, theta, alpha, iterations)
    print('Theta found by Gradient Descent: weight0 = {} , weight1 = {} , weight2 = {}'.format(theta[0], theta[1],
                                                                                               theta[2]))
    plot_j_hist(j_hist, iterations)


def func(X, theta):
    y = []
    x_1 = []
    x_2 = []
    for i in range(len(x)):
        res = X['house_size'][i] * theta[1] + X['room_number'][i] * theta[2] + theta[0]
        y.append(float(res))
    for i in range(len(x)):
        x_1.append(X['house_size'][i])
        x_2.append(X['room_number'][i])
    return x_1, x_2, y


def test():
    theta = np.zeros((3, 1))
    iterations = 1500
    alpha = 0.01
    j_hist, theta = gradientDescentMulti(x_norm, Y, theta, alpha, iterations)
    print('Theta found by Gradient Descent: weight0 = {} , weight1 = {} , weight2 = {}'.format(theta[0], theta[1],
                                                                                               theta[2]))
    ax = plt.axes(projection='3d')
    ax.scatter3D(X['house_size'], X['room_number'], Y)
    x_1, x_2, y_p = func(X, theta)
    ax.plot(x_1, x_2, y_p)
    plt.show()


main()
