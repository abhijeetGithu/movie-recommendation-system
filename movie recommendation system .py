#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np


# In[18]:


movies=pd.read_csv("tmdb_5000_movies.csv")
data=pd.read_csv("tmdb_5000_credits.csv")


# In[19]:


movies=movies.merge(data,on='title')


# In[20]:


movies


# In[22]:


movies.shape


# In[24]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[36]:


movies.head()


# In[29]:


import ast


# In[30]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[31]:


movies.dropna(inplace=True)


# In[33]:


movies.isnull().sum()


# In[48]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[49]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[50]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[52]:


movies['cast'][0]


# In[55]:


def convert3(text):
    L = []
    count=0
    for i in ast.literal_eval(text):
        if count != 3:
            L.append(i['name']) 
            count+=1
        else:
            break
    return L 


# In[56]:


movies['cast'] = movies['cast'].apply(convert3)
movies.head()


# In[59]:


movies.head()


# In[60]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[61]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[62]:


movies


# In[76]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[77]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[78]:


movies.head()


# In[79]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[91]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])


# In[97]:


new


# In[93]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))

new.head()


# In[89]:


vector.shape


# In[99]:


new


# In[100]:


new['tags']=new['tags'].apply(lambda x:x.lower())


# In[101]:





# In[106]:





# In[114]:





# In[ ]:





# In[118]:


import nltk


# In[119]:


from nltk.stem.porter import PorterStemmer


# In[120]:


ps=PorterStemmer()


# In[123]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)  


# In[125]:


new['tags']=new['tags'].apply(stem)


# In[126]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[127]:


vectors=cv.fit_transform(new['tags']).toarray()


# In[128]:


cv.get_feature_names()


# In[131]:


from sklearn.metrics.pairwise import cosine_similarity


# In[134]:


similarity=cosine_similarity(vectors)


# In[142]:


similarity[2]


# In[148]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[149]:


def recommend(movie):
    movie_index=new[new['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new.iloc[i[0]].title)
    


# In[150]:


recommend('Avatar')


# In[151]:


import pickle


# In[152]:


pickle.dump(new,open('movies.pkl','wb'))


# In[154]:


pickle.dump(new.to_dict(),open('movie_dict.pkl','wb'))


# In[155]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




