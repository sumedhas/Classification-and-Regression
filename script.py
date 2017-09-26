import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD
    labels = np.unique(y)
    N = len(X)

    # Calculating class-wise mean
    classMeans = np.zeros((5, 2))
    class_n = np.zeros((5, 1))

    for j in range(0, len(labels)):
        for i in range(0, len(X)):
            if (y[i] == labels[j]):
                classMeans[j] = classMeans[j] + X[i]
                class_n[j] = class_n[j] + 1
        classMeans[j] = classMeans[j] / class_n[j]
    means = classMeans.T

    # Calculating Covariance for all classes
    covmat = np.cov(X.T)

    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    labels = np.unique(y)
    N = len(X)

    # Calculating class-wise mean
    classMeans = np.zeros((5, 2))

    # Calculating mean for all classes
    class_n = np.zeros((5, 1))

    for j in range(0, len(labels)):
        for i in range(0, len(X)):
            if (y[i] == labels[j]):
                classMeans[j] = classMeans[j] + X[i]
                class_n[j] = class_n[j] + 1
        classMeans[j] = classMeans[j] / class_n[j]
    means = classMeans.T

    # Calculating Covariance for all classes
    covmats = []

    for j in range(0, len(labels)):
        xTemp = []
        for i in range(0, len(X)):
            if (y[i] == labels[j]):
                xTemp.append(X[i])
        np.asarray
        classCov = np.cov(np.asarray(xTemp).T)
        covmats.append(classCov)

    return means, covmats


def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    means = means.T
    allPdf = []
    ypred = None
     
    for j in range(0, len(Xtest)):
        for i in range(0, len(means)):                    
            correctedMean = np.subtract(Xtest[j],means[i])
            numerator = -(np.dot(np.dot(correctedMean, np.linalg.inv(covmat)), correctedMean.T)) * 0.5
            denominator = np.power((2 * np.pi), len(means)/2) * np.power(np.linalg.det(covmat),0.5)
            pdf = np.exp(numerator)/denominator            
            allPdf.append(pdf)
        max_pdf = max(allPdf)
        if ypred is None:
            ypred = np.array(float(allPdf.index(max_pdf) + 1))
        else:
            ypred = np.vstack((ypred, float(allPdf.index(max_pdf) + 1)))
        allPdf = []
    acc = np.mean(ypred == ytest)*100
    return acc, ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
      
    means = means.T
    allPdf = []
    ypred = None
     
    for j in range(0, len(Xtest)):
        for i in range(0, len(means)):                    
            correctedMean = np.subtract(Xtest[j],means[i])
            numerator = -(np.dot(np.dot(correctedMean, np.linalg.inv(covmats[i])), correctedMean.T)) * 0.5
            denominator = np.power((2 * np.pi), len(means)/2) * np.power(np.linalg.det(covmats[i]),0.5)
            pdf = np.exp(numerator)/denominator            
            allPdf.append(pdf)
        max_pdf = max(allPdf)
        if ypred is None:
            ypred = np.array(float(allPdf.index(max_pdf) + 1))
        else:
            ypred = np.vstack((ypred, float(allPdf.index(max_pdf) + 1)))            
        allPdf = []
    acc = np.mean(ypred == ytest)*100
    return acc, ypred

# problem 2
def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    # refer to slide linear regression p 10

    # wmap = (x'x)^-1 x' y
    w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return w

# problem 3
def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    # refer to slide linear regresion p 20

    # w map = (lambda Id + x' x)^-1 x' y
    w = np.dot(np.linalg.inv((lambd * np.identity(X.shape[1])) + (np.dot(X.T, X))), np.dot(X.T, y))

    return w

# problem 2
def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    # IMPLEMENT THIS METHOD
    # refer to pdf formula(3)

    # MSE = 1/N * sum of (yi - w'x)^2
    n = Xtest.shape[0]
    mse = np.sum(np.square((ytest - np.dot(Xtest, w)))) / n

    return mse

# problem 4
def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD

    # Error
    # refer to formula(4)
    error = (np.sum(np.square(y - np.dot(X, np.reshape(w, (w.size, 1)))) / 2)) + ((lambd * np.dot(w.T, np.reshape(w, (w.size, 1)))) / 2)

    # Error gradient with respect to w
    # refer to logistic regression slide p 8 "gradient descent for learning w"
    error_grad = (-1.0 * np.dot(y.T, X)) + (np.dot(np.dot(w.T, X.T), X)) + (lambd * w.T)

    return error.flatten(), error_grad.flatten()

def mapNonLinear(x, p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))

    # IMPLEMENT THIS METHOD
    Xd = np.ones((x.shape[0], p + 1))
    for i in range(p + 1):
        Xd[:, i] = pow(x, i)
    return Xd


# Main script

# Problem 1
print('problem 1')
# load the sample data
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')

# # LDA
means, covmat = ldaLearn(X, y)
ldaacc, ldares = ldaTest(means, covmat, Xtest, ytest)
print('LDA Accuracy = ' + str(ldaacc))
# QDA
means, covmats = qdaLearn(X, y)
qdaacc, qdares = qdaTest(means, covmats, Xtest, ytest)
print('QDA Accuracy = ' + str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5, 20, 100)
x2 = np.linspace(-5, 20, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.zeros((x1.shape[0] * x2.shape[0], 2))
xx[:, 0] = xx1.ravel()
xx[:, 1] = xx2.ravel()

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)

zacc, zldares = ldaTest(means, covmat, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zldares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc, zqdares = qdaTest(means, covmats, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zqdares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.title('QDA')

plt.show()

# Problem 2
print('problem 2')
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mle = testOLERegression(w, Xtest, ytest)
# training data
mle_learn_data = testOLERegression(w, X, y)

w_i = learnOLERegression(X_i, y)
mle_i = testOLERegression(w_i, Xtest_i, ytest)
# training data
mle_learn_data_intercept = testOLERegression(w_i, X_i, y)

print('MSE without intercept ' + str(mle))
print('MSE with intercept ' + str(mle_i))

print ('MSE training data without intercept '+str(mle_learn_data))
print ('MSE training data with intercept '+str(mle_learn_data_intercept))

# Problem 3
print('problem 3')
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k, 1))
mses3 = np.zeros((k, 1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i, y, lambd)
    mses3_train[i] = testOLERegression(w_l, X_i, y)
    mses3[i] = testOLERegression(w_l, Xtest_i, ytest)

    #print('lambda: ' + str(lambd) + ' Train: ' + str(mses3_train[i]) + ' Test: ' + str(mses3[i]))
    #print(str(mses3_train[i]))
    #print(str(mses3[i]))

    i = i + 1
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')

plt.show()

# # print w using OLE (with intercept)
# print('OLE')
# print(str(w_i))
# # print w using Ridge regression (with intercept)
# print('Ridge Regression')
# print(str(w_l))

# get optimal lambda
print('optimal lambda is: ' + str(lambdas[np.argmin(mses3)]))

# Problem 4
print('problem 4')
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k, 1))
mses4 = np.zeros((k, 1))
opts = {'maxiter': 100}  # Preferred value.
w_init = np.ones((X_i.shape[1], 1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l, [len(w_l), 1])
    mses4_train[i] = testOLERegression(w_l, X_i, y)
    mses4[i] = testOLERegression(w_l, Xtest_i, ytest)

    #print('lambda: '+ lambd + ' Train data: ' + mses4_train[i] + ' Test data: ' + mses4[i])
    #print(mses4_train[i])
    #print(mses4[i])

    i = i + 1
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses4_train)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses4)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])
plt.show()

# Problem 5
print('problem 5')
pmax = 7
lambda_opt = lambdas[np.argmin(mses3)]  # REPLACE THIS WITH lambda_opt estimated from Problem 3
print('optimal lambda is: ' + str(lambda_opt))
mses5_train = np.zeros((pmax, 2))
mses5 = np.zeros((pmax, 2))
for p in range(pmax):
    Xd = mapNonLinear(X[:, 2], p)
    Xdtest = mapNonLinear(Xtest[:, 2], p)
    #Lambda = 0
    w_d1 = learnRidgeRegression(Xd, y, 0)
    train_mse = testOLERegression(w_d1, Xd, y)
    mses5_train[p, 0] = train_mse
    test_mse = testOLERegression(w_d1, Xdtest, ytest)
    #print ('Lambda : 0 test_data_mse: '+str(test_mse))
    mses5[p, 0] = test_mse
    #Optimal Lambda 
    w_d2 = learnRidgeRegression(Xd, y, lambda_opt)
    train_mse_opt = testOLERegression(w_d2, Xd, y)
    test_mse_opt = testOLERegression(w_d2, Xdtest, ytest)
    mses5_train[p, 1] = train_mse_opt
    mses5[p, 1] = test_mse_opt
    

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax), mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization', 'Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax), mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization', 'Regularization'))
plt.show()