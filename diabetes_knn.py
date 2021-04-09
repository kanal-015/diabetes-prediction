#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[4]:


dataset=pd.read_csv('Downloads/diabetes (1).csv')
dataset.head()
dataset.columns


# In[5]:


#replace the value of zeros in all columns with mean of the whole data

replace_zeros=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI']

for column in replace_zeros:
    dataset[column]=dataset[column].replace(0,np.NaN)
    mean=int(dataset[column].mean(skipna=True))
    dataset[column]=dataset[column].replace(np.NaN,mean)
    
 


# In[6]:


#now let's split the dataset in train and test sets

x=dataset.iloc[:, 0:8] #here, [:,0:8] means-- (:) represents all rows and (0:8) represents from 0 till 7 columns. 0 to 7 columns will be in train set
y=dataset.iloc[:,8] #here, [:,8] means-- all rows and 8 means only last that is 9th column or 8th indiced column that is outcome will be our prediction so in test

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,test_size=0.3)


# In[7]:


#do scaling that is the values will be in different numbers that is 5, 9, 1, etc. we have to keep them in range of -1 to 1 so apply standard scalar function to keep all of them in range
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


# In[8]:


classifier=KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')


# In[9]:


classifier.fit(x_train,y_train)


# In[10]:


y_pred=classifier.predict(x_test)
y_pred


# In[11]:


conf_matx=confusion_matrix(y_test,y_pred)
print(conf_matx)


# In[12]:


print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[27]:


sns.countplot(dataset['Outcome'])


# In[29]:


# try K=1 through K=11 and record testing accuracy
k_range = range(1, 11)

# We can create Python dictionary using [] or dict()
scores = []

# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))

print(scores)


# In[30]:


# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# In[ ]:




