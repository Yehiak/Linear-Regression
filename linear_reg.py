import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'dataset.csv')
# print(df)
plt.scatter(df['customer_number'], df['profit'])
plt.xlabel("customer number")
plt.ylabel("profit")
plt.show()
x = df[['customer_number']]
y = df[['profit']]
# reshaping features and target data for entering the gradient function
xg = x.values.reshape(-1, 1)
yg = y.values.reshape(-1, 1)
# adding a column of 1's in the features matrix for the dot prouct with the weight matrix
xg = np.concatenate((np.ones(len(x)).reshape(-1, 1), x), axis=1)


# function for computing the cost(transfer) of the points where the error is calculated and returned
def computeCost(x, y, theta):
    n = len(y)
    h_x = x.dot(theta)
    j = np.sum(np.square(h_x - y)) * (1 / (2 * n))
    return j


# the main function where the update happens according to the direction of the gradient and weight update
def gradientDescent(x, y, theta, alpha, iterations):
    j_history = []  # a list for storing all the cost values to be plotted and used else where
    n = len(y)
    for i in range(iterations):
        j_history.append(computeCost(x, y, theta))
        h_x = x.dot(theta)
        theta = theta - ((alpha / n) * (np.dot(x.T, (h_x - y))))
        # The . T accesses the attribute T of the object, which happens to be a NumPy array. The T attribute is the
        # transpose of the array,
    return theta, j_history


# function for plotting the cost function values vs the iterations
def plot_j_hist(j_hist):
    plt.plot(np.arange(len(j_hist)), j_hist)
    plt.xlabel("Number of iterations (Epochs)")
    plt.ylabel("Cost function  J(Θ)")
    plt.title("Gradient Descent")
    plt.show()


# function for preparing the x and y_hat values to plot the MSE
def func(x, theta):
    y = []
    x_p = []
    for i in range(len(x)):
        res = x['customer_number'][i] * theta[1] + theta[0]
        y.append(float(res))
    for i in range(len(x)):
        x_p.append(x['customer_number'][i])
    return x_p, y


# the main function for applying all the other functions to work together
def main():
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    theta, j = gradientDescent(xg, yg, theta, alpha, iterations)
    print('Theta found by Gradient Descent: slope = {} and intercept {}'.format(theta[1], theta[0]))
    plot_j_hist(j)
    x_p, y_p = func(x, theta)
    plt.scatter(x, y, marker='o', color='green')
    plt.plot(x_p, y_p)
    plt.show()
    print(predict(35000, theta))
    print(predict(70000, theta))


def test():
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    theta, j = gradientDescent(xg, yg, theta, alpha, iterations)
    print('Theta found by Gradient Descent: slope = {} and intercept {}'.format(theta[1], theta[0]))
    print(func(x, theta))
    x_p, y_p = func(x, theta)
    plt.scatter(x, y, marker='o', color='green')
    plt.plot(x_p, y_p)
    plt.show()


# prediction function
def predict(x, theta):
    x_p = x/10000
    pred = x_p * theta[1] + theta[0]
    return"prediction of {} is: {} \n so profit is: {} ".format(x, pred[0], pred[0]*10000)


main()
