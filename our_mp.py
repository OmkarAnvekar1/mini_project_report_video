#!/usr/bin/env python
# coding: utf-8

# **Import required libraries**

# In[1]:


import os
import IPython
import cv2 as cv
import numpy as np
import seaborn as sns
import pandas as pd 
from scipy.io import wavfile 
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Algorithm Implementation without using sklearn library**

# In[2]:


def Datacentering( X : np.ndarray ):   
    """
    Make mean of each dependent component of input matrix X  zero (0). 
    """
    means = np.mean( X, axis=1, keepdims=True )
    Datacentered = X - means
    return Datacentered 


# In[3]:


def Decomposeigen( X : np.ndarray ):  
    """
    Find eigne values and eigne values of covariance matrix of centered Data.
    """
    covarianceMatrix =  np.cov( X )
    try : 
        eigValues, eigVectors = np.linalg.eig( covarianceMatrix )
    except Exception as e : 
        print( "Error : ", e)
        return 

    return eigVectors , np.diag(eigValues),


# In[ ]:





# In[4]:


def Datawhitening( X : np.ndarray ) : 
    """
    Find linear transformation L s.t. L@X is uncorrelated matrix with variance identity matrix  
    """
    E, D = Decomposeigen( X )
    L = np.linalg.inv( D**(1/2) ) @ E.T 
    whitenedData   = L@X 
    return L, whitenedData 


# In[5]:


def g( u  : np.ndarray , method= 1 ):
    if method == 1 :    
        gu =  np.tanh( u )

    if method == 2 : 
        gu =  u * np.exp( -u**2/2 )
    
    return gu


# In[6]:


def g_derivative( u : np.ndarray, method= 1):
    if method == 1 : 
        grad =  1 - np.square( np.tanh(u) )
    
    if method == 2 : 
        grad = ( 1- u**2 )* np.exp( -u**2/2 )
    
    return grad  


# In[7]:


def Sourcesrecover( V , origData, Datawhitening, Filterwhiten, isImage=False ):

    # project whitened data onto independent components:
    S = np.matmul(V, Datawhitening )

    # compute unmixing matrix:
    W = np.matmul(V,Filterwhiten)

    # estimate the mean and standard deviation of the sources:
    S_mean = np.matmul(W, np.mean(origData, axis=1, keepdims=True))
    S_std = np.matmul(V, np.std(origData, axis=1, keepdims=True))

    # add the mean and standard deviation of the sources back in:
    S = S_std * S + S_mean
    
    if isImage : 
        return S 
    
    S = np.int16(S)
    return S


# In[8]:


def ICAfast( X : np.ndarray, N, M, C ): 

    np.random.seed(42)   # For reproducability of code

    W = np.zeros((N,C))    
    epsilon = 1e-10
    for p in range(0, C): 
        Wp = np.random.randn(N,1)
        iteration = 0 
        max_iterations = 200
        while iteration < max_iterations : 
            Wp =   (1./M) * ( X @ ( g ( Wp.T @ X ).T ) - g_derivative( Wp.T @ X ) @ np.ones( (M,1) ) * Wp ) 
            Wp -=  np.sum( [ np.dot(np.squeeze(Wp), W[:, k]) * np.reshape(W[:, k], (N, 1)) for k in range(p) ], axis=0)
            Wp = Wp / np.linalg.norm(Wp)
            W[ :, p ] = Wp.reshape(-1) 
            iteration += 1     
    V = W.T 
    return V 


# **Getting Independent Sound signals from mixed signals using ICA**

# In[9]:


import os
import numpy as np
from scipy.io import wavfile

def getSoundFiles(folderNa.me="C:\\Users\\omkar\\Downloads\\MiniProjectInput"):
    sound_signals = []
    sound_files = []
    
    for root, _, files in os.walk(folderName): 
        for sound_file in files: 
            path = os.path.join(root, sound_file)
            rate, mixed_signal = wavfile.read(path)
            sound_signals.append(mixed_signal)      
            sound_files.append(path)

    # Ensure all signals are of the same length by truncating to the shortest signal length
    n = min([len(x) for x in sound_signals])
    sound_signals = np.array([signal[:n] for signal in sound_signals])
    
    return sound_files, sound_signals

# Call the function to get the file paths and signals
file_paths, signals = getSoundFiles()

# Print the file paths
for path in file_paths:
    print(path)

# Now you can use the 'signals' variable which contains the audio signals from all the files


# In[ ]:





# In[10]:


'''def plotSignals(sound_signals): 
    for j in range(len(sound_signals)):
        n = len(sound_signals[0])
        plt.figure(figsize=(12,2))
        plt.title('Recording %d'%(j+1))
        plt.plot(range(n), sound_signals[j,:], c="#3ABFE7")
        plt.ylim(-35000, 35000)
def showAudio(sound_files): 
    for i in range(len(sound_files)): 
        print("Audio %d"%(i+1))
        IPython.display.display(IPython.display.Audio(sound_files[i]))
def writeOutput( S , path="recovered_files/sound_seperated_"): 
    output = [ path +str(i+1)+".wav" for i in range(len(S)) ]
    for i in range( len (output )): 
        wavfile.write(output[i], rate=32000, data=S[i,:])
    return output
def Algorithm( sound_signals, returnUnmixingMatrix = False, isImage=False): 
    origData = sound_signals    # X is input signal matrix of shape N x M 
    N = origData.shape[0]       # Number of columns corresponding with the number of samples of in each mixed signals 
    M = origData.shape[1]       # Number of rows corresponding with the number of independent source signals
    C = origData.shape[0]       # Number of desired independent components 

    centeredData = Datacentering(origData)
    whitenFilter, whitenedData = Datawhitening(centeredData)
    V = ICAfast( whitenedData ,N, M , C ) 
    S = Sourcesrecover(V, origData, whitenedData, whitenFilter, isImage)  # source signals 
    unmixingMatrix = V@whitenFilter 
    if returnUnmixingMatrix : 
        return unmixingMatrix, S 
    return S 
# Read sound files 
sound_files, sound_signals = getSoundFiles("sound_files")

# Visualize signals 
plotSignals(sound_signals)

# Perform ICA 
S = Algorithm(sound_signals )

# Visualize output 
plotSignals(S)

# Write output to WAV format file 
output_sound_files = writeOutput(S)'''


# In[ ]:





# In[ ]:





# **Implementation of ICA**

# In[11]:


import os
import IPython
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def plotSignals(sound_signals): 
    for j in range(len(sound_signals)):
        n = len(sound_signals[0])
        plt.figure(figsize=(12, 2))
        plt.title('Recording %d' % (j + 1))
        plt.plot(range(n), sound_signals[j, :], c="#3ABFE7")
        plt.ylim(-35000, 35000)

def showAudio(sound_files): 
    for i in range(len(sound_files)): 
        print("Audio %d" % (i + 1))
        IPython.display.display(IPython.display.Audio(sound_files[i]))

def writeOutput(S, path="C:\\Users\\omkar\\Downloads\\MiniProjectInput\\output"): 
    if not os.path.exists(path):
        os.makedirs(path)

    output = [os.path.join(path, "output_file" + str(i + 1) + ".wav") for i in range(len(S))]

    for i in range(len(output)): 
        wavfile.write(output[i], rate=32000, data=S[i, :])

    return output



# In[12]:


def Algorithm(sound_signals, returnUnmixingMatrix=False, isImage=False): 
    origData = sound_signals    # X is the input signal matrix of shape N x M 
    N = origData.shape[0]       # Number of columns corresponding to the number of samples in each mixed signal 
    M = origData.shape[1]       # Number of rows corresponding to the number of independent source signals
    C = origData.shape[0]       # Number of desired independent components 

    centeredData = Datacentering(origData)
    whitenFilter, whitenedData = Datawhitening(centeredData)
    V = ICAfast(whitenedData, N, M, C) 
    S = Sourcesrecover(V, origData, whitenedData, whitenFilter, isImage)  # source signals 
    unmixingMatrix = V @ whitenFilter 
    if returnUnmixingMatrix: 
        return unmixingMatrix, S 
    return S 





# In[13]:


# Read sound files 
sound_files, sound_signals = getSoundFiles("MiniProjectInput")

# Visualize signals 
plotSignals(sound_signals)


# **Perform ICA**

# In[14]:


# Perform ICA 
S = Algorithm(sound_signals)

# Visualize output 
plotSignals(S)

# Write output to WAV format file 
output_sound_files = writeOutput(S)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


import numpy as np

def Decomposeigen(X: np.ndarray):
    """
    Find eigenvalues and eigenvectors of the covariance matrix of centered Data.
    """
    covarianceMatrix = np.cov(X)
    try:
        eigValues, eigVectors = np.linalg.eig(covarianceMatrix)
        # Print eigenvalues and eigenvectors
        print("Eigenvalues:")
        print(eigValues)
        print("Eigenvectors:")
        print(eigVectors)
    except Exception as e:
        print("Error:", e)
        return

    return eigVectors, np.diag(eigValues)

# Example usage:
# Assuming X is your centered data matrix
X = np.random.rand(3, 3)  # Replace this with your actual data
result = Decomposeigen(X)


# In[ ]:




