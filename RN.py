#!/usr/bin/env python
# coding: utf-8

# In[246]:

import re
import matplotlib.pyplot as plt
import numpy as np
import re
import math
class Sigmoide:

    def adelante(self, x):
        return 1/(1+np.exp(x))

    def derivada(self, x):
        return self.adelante(x)*(1-self.adelante(x))


class Escalon:

    def adelante(self, x):
        if x < 0:
            print(0)
        elif x >= 0:
            return x*1

    def derivada(self, x):
        return x*0


class Tangh:

    def adelante(self, x):
        return np.exp(x)-np.exp(-x)/(np.exp(x)+np.exp(-x))

    def derivada(self, x):
        return 1-np.power(self.adelante(x), 2)


# In[247]:


def normalize(X, nh=1, nl=0):
    X=np.array(X,float)
    try:
        cols=X.shape[1]
    except:
        print('no es una matriz')
        return X
    for i in range(0,cols):  
        dh=max(X[:,i])
        dl=min(X[:,i])
        X[:,i]=np.subtract(X[:,i],dl) *(nh - nl)/ (dh-dl)
    return X


def hotencoding(Y):  # Asumiendo que viene con 1,2,3,4,5 de clase.
    clases = dict()
    l=len(np.unique(Y))
    n = 0
    for i in np.unique(Y):
        aux=np.zeros(l)
        aux[n]=1
        clases[i]=aux
        n+=1
    Ya=[]
    for j in range(0,len(Y)):
        Ya.append(clases[Y[j]])
    return np.array(Ya), clases


# In[1509]:





archivo = open('seeds_dataset.txt', 'r')
data = archivo.read()
archivo.close()
data = re.sub('\t|\n', ' ', data).split()
data = [float(n) for n in data ]
data = np.reshape(data, [210, 8])
Ypre = data[:, -1].T
Xpre = data[:, :-1].T


# In[1498]:



def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Produce a neural network randomly initialized
def initialize_parameters(n_x, n_hl,n_h, n_y):
    if not n_hl==len(n_h):
        print('error, el número de n_hl debe ser consistente con la cantidad de elementos en n_h')
    #Se agrega un parámetro que se llam n_hl, que sirve para poder agregar una cantidad de capas ocultas a elección.
    parameters = dict()
    parameters['Wx'] = np.random.randn(n_h[0], n_x)
    parameters['bx'] = np.zeros((n_h[0], 1))
    lista = ['x']
    for i in range(1,n_hl):
        nW='W{}'.format(i)
        nb='b{}'.format(i)
        parameters[nW]=np.random.randn(n_h[i], n_h[i-1])
        parameters[nb]=np.zeros((n_h[i], 1))
        lista.append(i)        
    parameters['Wy'] = np.random.randn(n_y, n_h[-1])
    parameters['by'] = np.zeros((n_y, 1))
    lista.append('y')
    parameters['lista']=lista
    
    return parameters
    

# Evaluate the neural network
def forward_prop(X, parameters):
    pK = parameters.keys()
    Wi = [W for W in pK if re.match(r'W[0-9]+',W)]
    bi = [b for b in pK if re.match(r'b[0-9]+',b)]
    i  = parameters['lista']
    n_hl=len(Wi)+1
    cache=dict()
    
    Zx=np.dot(parameters['Wx'], X) + parameters['bx']
    cache['Ax']=np.tanh(Zx)
    for j in range(1,len(i)):
        nA1='A{}'.format(i[j-1])
        nA='A{}'.format(i[j])
        Zi= np.dot(parameters['W{}'.format(i[j])], cache[nA1]) + parameters['b{}'.format(i[j])]
        cache[nA]=np.tanh(Zi)

    
    return cache[nA], cache

# Evaluate the error (i.e., cost) between the prediction made in A2 and the provided labels Y 
# We use the Mean Square Error cost function
def calculate_cost(A2, Y):
    # m is the number of examples
    cost = np.sum((0.5 * (A2 - Y) ** 2).mean(axis=1))/m
    return cost

# Apply the backpropagation
def backward_prop(X, Y, cache, parameters):
    pK = parameters.keys()
    Wi = [W for W in pK if re.match(r'W[0-9]+',W)]
    bi = [b for b in pK if re.match(r'b[0-9]+',b)]
    i  = parameters['lista']
    n_hl=len(Wi)+1
    grads=dict()
    
    dZac=cache['Ay']- Y
    lac='y'
    prox=i[len(i)-2]
    grads['dW{}'.format(lac)] = np.dot(dZac, cache['A{}'.format(prox)].T)/m
    grads['db{}'.format(lac)] = np.sum(dZac, axis=1, keepdims=True)/m  
    
    for j in range(2,len(i)):
        J=len(i)-j
        lan=lac
        lac=i[J]
        prox=i[J-1]
        dZan=dZac
        dZac = np.multiply(np.dot(parameters['W{}'.format(lan)].T, dZan), 1-np.power(cache['A{}'.format(lac)], 2))
        grads['dW{}'.format(lac)] = np.dot(dZac, cache['A{}'.format(prox)].T)/m
        grads['db{}'.format(lac)] = np.sum(dZac, axis=1, keepdims=True)/m    

    dZan=dZac
    lan=lac
    dZx = np.multiply(np.dot(parameters['W{}'.format(lan)].T, dZan), 1-np.power(cache['A{}'.format('x')], 2))
    grads['dWx'] = np.dot(dZx, X.T)/m
    grads['dbx'] = np.sum(dZx, axis=1, keepdims=True)/m    

    
    return grads

# Third phase of the learning algorithm: update the weights and bias
def update_parameters(parameters, grads, learning_rate):
    
    
    pK = parameters.keys()
    Wi = [W for W in pK if re.match(r'W[0-9]+',W)]
    bi = [b for b in pK if re.match(r'b[0-9]+',b)]
    i  = parameters['lista']
    n_hl=len(Wi)+1
    new_parameters=dict()
    
    for l in i:
        Wi=parameters['W{}'.format(l)]
        bi=parameters['b{}'.format(l)]
        dWi = grads["dW{}".format(l)]
        dbi = grads["db{}".format(l)]
        Wi = Wi - learning_rate*dWi
        bi = bi - learning_rate*dbi
        new_parameters['W{}'.format(l)] = Wi
        new_parameters['b{}'.format(l)] = bi
    new_parameters['lista']=i

    return new_parameters

# model is the main function to train a model
# X: is the set of training inputs
# Y: is the set of training outputs
# n_x: number of inputs (this value impacts how X is shaped)
# n_h: number of neurons in the hidden layer
# n_y: number of neurons in the output layer (this value impacts how Y is shaped)
def model(X, Y, n_x, n_hl, n_h, n_y, num_of_iters, learning_rate,norm=True):
    #if norm:
        #X=normalize(X)
    parameters = initialize_parameters(n_x, n_hl, n_h, n_y)
    errs=[]
    yes =[]
    for i in range(0, num_of_iters+1):
        a2, cache = forward_prop(X, parameters)
        cost = calculate_cost(a2, Y)
        grads = backward_prop(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)
        errs.append(cost)
        yes.append(a2)
        
        if(i%100 == 0):
            print('Cost after iteration# {:d}: {:f}'.format(i, cost))

    return parameters, errs, yes
 
# Make a prediction
# X: represents the inputs
# parameters: represents a model
# the result is the prediction
def predict(X, parameters):
    a2, cache = forward_prop(X, parameters)
    yhat = a2
    yhat = np.squeeze(yhat)
    y_predict=np.zeros(yhat.shape)
    nY=Y.T.shape[0]
    try:
        for i in range(yhat.shape[1]):
            aux=np.zeros(nY)
            indP=np.where(yhat[:,i]==max(yhat[:,i]))[0]
            aux[indP]=1
            y_predict[:,i]=aux
    except:
        y_predict=yhat
        #for i in range(yhat):
            #aux=np.zeros(nY)
            #indP=np.where(yhat[i]==max(yhat[i]))[0]
            #aux[indP]=1
            #y_predict[:,i]=aux
            #y_predict=yhat
        
    return y_predict



def muestraconfusion(Confusion):
    head=['\t']
    for i in range(Confusion.shape[0]):
        head.append('C{}\t'.format(i))

    H=''
    for i in head:
           H+=i
    H+='\n'
    for i in range(Confusion.shape[0]):
        a=head[i+1]
        for j in range(Confusion.shape[0]):
            a+='%.2f\t'%(Confusion[i,j])
        a+='\n'
        H+=a
    print(H)
    return(H)


def TestTrain(Xd,Yd,propE=0.8,semilla=42):
    np.random.seed(semilla)
    dX=Xd.shape[0]
    N=Xd.shape[1]
    To=np.concatenate((Xd,Yd))
    np.random.shuffle(To.T)
    NE= int(np.floor(N*propE))
    XE=To[:dX,:NE]
    YE=To[dX:,:NE]
    XT=To[:dX,NE:]
    YT=To[dX:,NE:]

    return XE,YE,XT,YT



def evaluate(Y_pre,Y):
    nY=Y.T.shape[1]
    Confusion=np.zeros((nY,nY))
    for i in range(Y.T.shape[0]):
        aux=np.zeros(nY)
        indP=np.where(Y_pre[:,i]==1)[0][0]
        indC=np.where(Y[:,i]==1)[0]
        Confusion[indP,indC]+=1
        
        aux[indP]=1
        Y_pre[:,i]=aux
    
    return Confusion

def Kfold(Xd,Yd,n_x, n_hl ,n_h, n_y, num_of_iters, learning_rate,k=5,semilla=42):

    dX=Xd.shape[0]
    N=Xd.shape[1]
    To=np.concatenate((Xd,Yd))
    np.random.shuffle(To.T)
    
    n=int(np.floor(N/k))
    sets=dict()
    for i in range(k-1):

        sets['X{}'.format(i)]=To[:dX,(i)*n:n*(i+1)]
        sets['Y{}'.format(i)]=To[dX:,(i)*n:n*(i+1)]
    i+=1
    sets['X{}'.format(i)]=To[:dX,(i)*n:]
    sets['Y{}'.format(i)]=To[dX:,(i)*n:]


    ky=[]

    for i in sets.keys():
        ky.append(i)
    for i in sets.keys():
        ky.append(i)
    kx=[w for w in ks if re.match(r'X[\d]+',w)]
    ky=[w for w in ks if re.match(r'Y[\d]+',w)]
    n=0
    Confus=dict()
    Yiter=dict()
    for i in range(k): 
        kn=n+k
        XE=sets[kx[n]].T
        for x in kx[n+1:kn-1]:
            XE=np.concatenate((XE,sets[x].T))
        YE=sets[ky[n]].T
        for x in ky[n+1:kn-1]:
            YE=np.concatenate((YE,sets[x].T))
        XT=sets[kx[n]].T
        for x in kx[n+1:kn-1]:
            XT=np.concatenate((XT,sets[x].T))
        YT=sets[ky[n]].T
        for x in ky[n+1:kn-1]:
            YT=np.concatenate((YT,sets[x].T))
        n+=1

        trained_parameters, errs, yes = model(XE.T, YE.T, n_x, n_hl ,n_h, n_y, num_of_iters, learning_rate)
        Y_predict=predict(XT.T, trained_parameters)
        Yiter[i]=Y_predict
        Confus[i]=evaluate(Y_predict,YT.T)

    ac=np.zeros(Confus[i].shape)    
    for i in range(k):
        ac=ac + Confus[i]
    muestraconfusion(ac/k)
    return Confus,Yiter


# In[1480]:


# Set the seed to make result reproducible
np.random.seed(42)

# The 4 training examples by columns
#X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

# The outputs of the XOR for every example in X
#Y=Ypre
Y ,clas = hotencoding(Ypre)

# No. of training examples
X=normalize(Xpre.T)
#X=Xpre
#Xpre)
m = X.T.shape[1]
#m = X.shape[1]

# Set the hyperparameters
n_x = 7     #No. of neurons in first layer
n_h = [2]     #No. of neurons in hidden layer
n_y = 3    #No. of neurons in output layer
n_hl= 1
num_of_iters = 10000
learning_rate = 0.01

# define a model 



XE,YE,XT,YT=TestTrain(X.T,Y.T)



# In[1488]:




trained_parameters = model(XE, YE, n_x, n_hl ,n_h, n_y, num_of_iters, learning_rate)
#trained_parameters, errs,yes = model(X.T, Y.T, n_x, n_hl ,n_h, n_y, num_of_iters, learning_rate)
Y_predict=predict(XT, trained_parameters)
Confusion=evaluate(Y_predict,YT)
muestraconfusion(Confusion)


# In[1514]:

np.random.seed(42)
trained_parameters, errs, yes= model(XE, YE, n_x, n_hl ,n_h, n_y, num_of_iters, learning_rate)


Y_predict=predict(XT, trained_parameters)
Confusion=evaluate(Y_predict,YT)
muestraconfusion(Confusion)

plt.plot(errs)
plt.xlabel='Epocas'
plt.ylabel='Error'
plt.show()



np.random.seed(42)

confus,yiter=Kfold(X.T, Y.T,n_x, n_hl ,n_h, n_y, num_of_iters, learning_rate,k=5,semilla=42)

