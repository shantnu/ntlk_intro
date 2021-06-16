#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet


# In[2]:


stopwords.words('english')[:16]


# In[3]:


# https://en.wikipedia.org/wiki/Cadet_Nurse_Corps
para = "The program was open to all women between the ages of 17 and 35, in good health, who had graduated from an accredited high school. Successful applicants were eligible for a government subsidy, paying for tuition, books, uniforms, and a stipend. In exchange, they were required to pledge to actively serve in essential civilian or federal government services for the duration of World War II. All state nursing schools were eligible to participate in the program. However, they needed to be accredited by the accrediting agency in their state, and connected with a hospital that had been approved by the American College of Surgeons."
words = word_tokenize(para)
print(words)
useful_words = [word for word in words if word not in stopwords.words('english')]
print(useful_words)


# In[4]:


movie_reviews.words()


# In[5]:


movie_reviews.categories()


# In[6]:


movie_reviews.fileids()[:4]


# In[7]:


all_words = movie_reviews.words()

freq_dist = nltk.FreqDist(all_words)

freq_dist.most_common(20)


# In[ ]:




