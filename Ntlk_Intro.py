#!/usr/bin/env python
# coding: utf-8

# Ntlk book is here: http://www.nltk.org/book/
# 
# NLP applications: http://blog.mashape.com/list-of-25-natural-language-processing-apis/

# In[6]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet


# In[7]:


sentence = "The Quick brown fox, Jumps over the lazy little dog. Hello World."


# In[8]:


sentence.split(" ")


# In[10]:


word_tokenize(sentence)


# In[11]:


w = word_tokenize(sentence)
nltk.pos_tag(w)


# In[12]:


# List of tages: http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

nltk.help.upenn_tagset()


# In[13]:


syn = wordnet.synsets("computer")
print(syn)
print(syn[0].name())
print(syn[0].definition())

print(syn[1].name())
print(syn[1].definition())


# In[14]:


syn = wordnet.synsets("talk")
syn[0].examples()


# In[15]:


syn = wordnet.synsets("speak")[0]
print(syn.hypernyms())
print(syn.hyponyms())


# In[16]:


syn = wordnet.synsets("good")
for s in syn:
    for l in s.lemmas():
        if (l.antonyms()):
            print(l.antonyms())


# In[17]:


syn = wordnet.synsets("book")
for s in syn:
    print(s.lemmas())

