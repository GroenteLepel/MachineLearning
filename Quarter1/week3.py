#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np
import matplotlib.pyplot as plt


# ## The excersize

# Write a computer program that implements the perceptron learning rule. Take as data p random input vectors of dimension n with binary components. Take as outputs random assignments \pm 1. Take n=50 and test empirically that when p < 2 n the rule converges almost always and for p > 2n the rule converges almost never.
# Reconstruct the curve C(p,n) as a function of p for n=50 in the following way. For each p construct a number of learning problems randomly and compute the fraction of these problems for which the perceptron learning rule converges. Plot this fraction versus p.

# In[76]:


p=200 #The number of points
n=50 #The number of dimensions
x=np.random.choice([-1,1],(p,n))  #This makes a p by n. Each row is a point
t=np.random.choice([-1,1],(p,1))  #This represents the label


# In[67]:


w=np.random.uniform(-0.5,0.5,(n,1)) #This makes a random weight initialization
eta=0.45 #This sets the learning rate


# In[68]:


count=0 #I keep count to make sure the while loop stops.
maxcount=50000 #The max count we allow
while np.sum(np.sign((x.dot(w)))*t)<p: #If all the points are classified correctly this sum should be equal to p
    for j in range(0,p):  #We go trough all the points
        dw=np.random.uniform(-0.0,0.0,(n,1)) #This is just an eleborate way to make an empty array, could surely be done nicer
        for i in range(0,n): #We update all the components of w
            if np.sum(np.sign((x.dot(w)))*t)<p:
                dw[i]=dw[i]+eta*np.heaviside(-np.transpose(w).dot(x[j]*t[j]),0)*x[j,i]*t[j]
            count=count+1
        w=np.add(w,dw)
    if count>maxcount: #If we haven't reached convergence after 50.000 iterations, we give up.
        break

if count<maxcount:
    print('convergence achieved')
if count>maxcount:
    print('convergence failed')


# In[59]:


np.sum(np.sign((x*t).dot(w)))


# Part 2 of the excercise: Reconstruct the curve C(p,n) as a function of p for n=50 in the following way. For each p construct a number of learning problems randomly and compute the fraction of these problems for which the perceptron learning rule converges. Plot this fraction versus p.

# In[71]:


def C(n,p):
    x=np.random.choice([-1,1],(p,n)) 
    t=np.random.choice([-1,1],(p,1))
    w=np.random.uniform(-0.5,0.5,(n,1)) #This makes a random weight initialization
    eta=0.45 #This sets the learning rate
    count=0 #I keep count to make sure the while loop stops.
    maxcount=50000 #The max count we allow
    while np.sum(np.sign((x.dot(w)))*t)<p: #If all the points are classified correctly this sum should be equal to p
        for j in range(0,p):  #We go trough all the points
            dw=np.random.uniform(-0.0,0.0,(n,1)) #This is just an eleborate way to make an empty array, could surely be done nicer
            for i in range(0,n): #We update all the components of w
                if np.sum(np.sign((x.dot(w)))*t)<p:
                    dw[i]=dw[i]+eta*np.heaviside(-np.transpose(w).dot(x[j]*t[j]),0)*x[j,i]*t[j]
                count=count+1
            w=np.add(w,dw)
        if count>maxcount: #If we haven't reached convergence after 50.000 iterations, we give up.
            break

    if count<maxcount:
        return(1)
    if count>maxcount:
        return(0)
    


# We wrote a function that for a given n and p returns a 1 is the problem was linearly seperable and 0 if not

# In[97]:


allp=np.linspace(1,200,10, dtype = int, endpoint=True)
D=15
succes=np.zeros(len(allp))
for l in range(0,len(allp)):
    for d in range(0,D):
        succes[l]=succes[l]+C(50,allp[l])


# In[98]:


plt.plot(allp,succes/D)
plt.xlabel('p')
plt.ylabel('C(n,p)/2^p')
plt.show()


# In[86]:


np.linspace(1,20,5, dtype = int, endpoint=True)


# In[ ]:




