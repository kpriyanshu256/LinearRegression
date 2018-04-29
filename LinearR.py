import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

def readData(file):
    data=pd.read_csv(file,names=[0,1])
    data = data.sample(frac=1).reset_index(drop=True)
    data=(data-data.mean())/data.std()
    X=data.iloc[:,0]
    y=data.iloc[:,1]
    o=pd.DataFrame(np.ones(shape=(X.shape[0],1)))
    X=pd.concat([o,X],axis=1)
    X=np.array(X)
    y=np.array(y)
    m,n=X.shape
    y=y.reshape(m,1)
    return X,y

def split_data(X,y,ratio):
    k=ratio*(X.shape[0])
    k=int(k)
    X_train=X[0:k,:]
    y_train=y[0:k,:]
    X_test=X[k:,:]
    y_test=y[k:,:]
    return X_train,y_train,X_test,y_test

class LinearR:
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.W=np.zeros(shape=(2,1))
        self.m=X.shape[1]
    
    def predict(self,X_test):
        #hypothesis
        return np.dot(X_test ,self.W)
    
    def cost(self):
        #calculate cost function
        return (np.sum((self.predict(self.X)-self.y)**2))/(2*self.m)

    def optimize(self):
        #minimize cost function
        epoch=1500
        alpha=0.01
        for i in range(epoch):
            grad=(np.dot(self.X.T,self.predict(self.X)-self.y))
            self.W=self.W-(alpha/self.m)*grad
        
            
    def results(self,X_test,y_test):
        plt.figure(2)
        red=mpatches.Patch(color='red', label='Training data')
        blue=mpatches.Patch(color='blue', label='Testing data')
        green=mpatches.Patch(color='green', label='Predicted')
        #training plot
        plt.scatter(self.X[:,1],self.y,color='red')
        #testing plot
        plt.scatter(X_test[:,1],y_test,color='blue')
        h=self.predict(X_test)
        plt.plot(X_test[:,1],h,color='green')
        plt.legend(handles=[red,blue,green])
        plt.show()
        
        
if __name__=='__main__':
    file='data/ex1data1.txt'
    X,y=readData(file)
    X_train,y_train,X_test,y_test=split_data(X,y,0.3)
    o=LinearR(X_train,y_train)
    o.optimize()
    o.results(X_test,y_test)