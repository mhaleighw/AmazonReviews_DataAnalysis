#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import sqlite3


# In[3]:


connection = sqlite3.connect(r'/Users/haleigh/Desktop/Udemy Courses/Data Analysis Projects/Amazon Customers Project/database.sqlite')


# In[4]:


type(connection)


# In[5]:


df = pd.read_sql_query("SELECT * FROM REVIEWS", connection)


# In[6]:


# raw data
df.shape


# In[7]:


# data preparation for analysis
df.columns


# In[8]:


df[df['HelpfulnessNumerator'] > df['HelpfulnessDenominator']]


# In[9]:


# valid rows
df_valid = df[df['HelpfulnessNumerator'] <= df['HelpfulnessDenominator']]


# In[10]:


df_valid.shape


# In[11]:


df_valid.columns


# In[12]:


# remove duplicate rows for unbiased results
df_valid.duplicated(['UserId', 'ProfileName','Time','Text'])


# In[13]:


# shows count of duplicate rows
df_valid[df_valid.duplicated(['UserId', 'ProfileName','Time','Text'])]


# In[14]:


# remove duplicate rows (174521 rows × 10 columns)
data = df_valid.drop_duplicates(subset =['UserId', 'ProfileName','Time','Text'])


# In[15]:


data.shape


# In[16]:


data.dtypes


# In[17]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[18]:


data['Time'] = pd.to_datetime(data['Time'], unit= 's')


# In[19]:


# analyse what amazon can recommend more to a user
data['ProfileName']


# In[20]:


data['ProfileName'].unique() # get unique names


# In[21]:


data['UserId'].nunique() # count of unique users


# In[22]:


recommend_df = data.groupby(['UserId']).agg({'Summary':'count', 'Text':'count', 'Score':'mean', 'ProductId':'count'}).sort_values(by='ProductId',ascending=False)


# In[23]:


recommend_df.columns = ['Number_of_Summaries', 'Num_Text', 'Average_Score', 'Products_Purchased']


# In[24]:


recommend_df


# In[25]:


recommend_df.index[0:10]


# In[26]:


recommend_df['Products_Purchased'][0:10].values


# In[27]:


plt.bar(recommend_df.index[0:10], recommend_df['Products_Purchased'][0:10].values)
plt.xticks(rotation='vertical')


# In[28]:


# which product has a good amount of reviews
# how many unique products do we have in data?
len(data['ProductId'].unique())


# In[29]:


# threshold value for a "good amount" of products should be greater than 500
product_count = data['ProductId'].value_counts().to_frame()


# In[30]:


product_count[product_count['ProductId']>500]


# In[31]:


# most frequent products
frequent_product_ids = product_count[product_count['ProductId']>500].index 


# In[32]:


data['ProductId'].isin(frequent_product_ids)


# In[33]:


frequent_product_df = data[data['ProductId'].isin(frequent_product_ids)]


# In[34]:


frequent_product_df.columns


# In[35]:


sns.countplot(y = 'ProductId', data = frequent_product_df, hue = 'Score')


# In[36]:


# is there a difference between the behabior of the freq. viewers and not freq. viewers regarding a purchase
# freq. viewer = bought the product 50 times or more


# In[37]:


x = data['UserId'].value_counts()


# In[38]:


x['AY12DBB0U420B']


# In[39]:


# if a user has a count of 50 it will be frequent
# consider the user as a pointer to each row of the UserId
data['viewer_type'] = data['UserId'].apply(lambda user : "Frequent" if x[user]>50 else "Not Frequent")


# In[40]:


data.head(3)


# In[41]:


data['viewer_type'].unique()


# In[42]:


not_frequent_df = data[data['viewer_type'] == 'Not Frequent']
frequent_df = data[data['viewer_type'] == 'Frequent']


# In[43]:


frequent_df['Score'].value_counts().plot(kind='bar')


# In[44]:


not_frequent_df['Score'].value_counts().plot(kind='bar')


# In[45]:


# are frequent viewers more likely to leave reviews?
data[['UserId', 'ProductId', 'Text']]


# In[46]:


def calculate_length(text):
    return len(text.split(' '))


# In[47]:


data['Text_length'] = data['Text'].apply(calculate_length)


# In[48]:


data['viewer_type'].unique()


# In[49]:


not_frequent_data = data[data['viewer_type'] == 'Not Frequent']
frequent_data = data[data['viewer_type'] == 'Frequent']


# In[50]:


fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.boxplot(frequent_data['Text_length'])
ax1.set_xlabel('Frequency of Reviewers')
ax1.set_ylim(0,600)

ax2 = fig.add_subplot(122)
ax2.boxplot(not_frequent_data['Text_length'])
ax2.set_xlabel('Non-Frequency of Reviewers')
ax2.set_ylim(0,600)


# In[51]:


# perform sentiment analysis on the data
get_ipython().system('pip install textblob')
from textblob import TextBlob


# In[53]:


data['Summary'][0]


# In[54]:


TextBlob('Good Quality Dog Food').sentiment.polarity


# In[56]:


sample = data[0:50000]


# In[57]:


polarity = []

for text in sample['Summary']:
    try:
        polarity.append(TextBlob(text).sentiment.polarity)
    except:
        polarity.append(0)


# In[58]:


len(polarity)


# In[59]:


sample['polarity'] = polarity


# In[61]:


# now we have a polarity feature
sample.head()


# In[64]:


# entire data frame for negative polarity
negative_polarity = sample[sample['polarity'] < 0]
positive_polarity = sample[sample['polarity'] > 0]


# In[66]:


from collections import Counter


# In[68]:


Counter(negative_polarity['Summary']).most_common(50)


# In[69]:


Counter(positive_polarity['Summary']).most_common(50)


# In[ ]:




