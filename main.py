#Mahmoud Mohamed Amr 20176027
#Hady Raed 20175019
import numpy as nump
import pandas as pd
header= ['exam1', 'exam2', 'adresult']
data = pd.read_csv('./admissionsScores.txt', names=(header))
X= data.values[0:80,0:2]
Y= data.values[0:80,2:3]
XTest= data.values[80:100,0:2]
YTest= data.values[80:100,2:3]

def normalization (X):
    return (X-nump.min(X, axis=0))/(nump.max(X, axis=0)-nump.min(X, axis=0))
XTest=normalization(XTest)    
X=normalization(X)
X = nump.hstack((nump.matrix(nump.ones(X.shape[0])).T, X))
XTest = nump.hstack((nump.matrix(nump.ones(XTest.shape[0])).T, XTest))
thetas = nump.zeros((1, X.shape[1]))
alpha= 0.0111
iterations= 1500

def hypothesis(thetas,X):
    theta = thetas.T
    z=nump.dot(X,theta)
    h= 1/ (1+ nump.exp(-z))
    return h

def gradientDescent(X, Y, thetas, alpha, iterations):
    cost_list = []
    for _ in range(0, iterations):
        cost_list.append(costFunc(X, Y, thetas))
        thetas = thetas - nump.multiply(alpha, nump.dot((hypothesis(thetas,X) - Y).T, X))/len(Y)
    return thetas  , cost_list


def costFunc(X, Y, thetas):
    part1= nump.multiply( (-Y) , nump.log10(hypothesis(thetas,X)))
    part2= nump.multiply((1-Y) , nump.log10(1-hypothesis(thetas, X)))
    cost = part1 - part2
    return cost

theta,cost_list =gradientDescent(X, Y, thetas, alpha, iterations)

def predict(theta,X):
    return hypothesis(theta, X)
    
def accuracyCalc(XTest,YTest,theta):
    counter=0
    p=predict(theta, XTest)
    for i in range (0,20):
        if(nump.equal(nump.round(p[i]), YTest[i])):
            counter=counter+1
    return counter/len(YTest) *100

calc= accuracyCalc(XTest, YTest, theta)
print(calc)
