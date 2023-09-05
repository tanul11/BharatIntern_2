#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


data = pd.read_csv("C:/Users/Tanul Tripathi/OneDrive/Desktop/datasets/Iris.csv")


# In[3]:


data.head()


# In[4]:


#getting insight of data
data.describe()


# In[5]:


data.info()


# In[7]:


data['Species'].value_counts()


# In[8]:


data.isnull().sum()


# In[9]:


data['SepalLengthCm'].hist()


# In[10]:


data['SepalWidthCm'].hist()


# In[11]:


data['PetalLengthCm'].hist()


# In[12]:


data['PetalWidthCm'].hist()


# In[16]:


#visualize the whole dataset
sns.pairplot(data, hue = 'Species')


# In[18]:


data.corr()


# In[29]:


cor = data.corr()
y = {"fontsize":15, "color":"r"}
fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(cor, annot = True, ax = ax, annot_kws = y, linewidth = 2, linecolor = "blue",cbar = False, cmap = 'twilight_shifted_r')
plt.show()


# In[30]:


#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[31]:


data['Species'] = le.fit_transform(data['Species'])
data.head()


# In[32]:


df = data.values
X = df[:,0:4]
Y = df[:,4]


# In[33]:


#Split the data to train and test dataset.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 0.2)


# In[34]:


model = LinearRegression()
model.fit(X_train, Y_train)


# In[ ]:


print('Accuracy :',model.score(X_))

