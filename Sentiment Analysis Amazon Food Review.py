#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Importing Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import nltk
nltk.download('averaged_perceptron_tagger')


# In[11]:


df = pd.read_csv(r'Reviews.csv')


# In[12]:


df = df.head(50)
df


# In[16]:


df['Text'].values[0]


# In[17]:


print(df.shape)


# In[18]:


Score_card = df['Score'].value_counts().sort_index()                     .plot(kind = 'bar', 
                          title = "Review by Stars",
                              figsize = (10, 5))
Score_card.set_xlabel('Review Stars')
plt.show()


# In[19]:


examp = df['Text'][15]
examp


# In[20]:


from nltk.tokenize import word_tokenize,sent_tokenize,wordpunct_tokenize
import nltk


# In[21]:


tokens = wordpunct_tokenize(examp)
tokens[:10]


# In[22]:


from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')

tagged = pos_tag(tokens)
tagged


# In[23]:


nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import chunk

entities = chunk.ne_chunk(tagged)
entities.pprint()


# # VADER - Sentiment Analysis

# In[24]:


#Bag of Words Approach
## 1. Stop words are removed
### 2.Each word is scored and combined to a total score

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


# In[25]:


sia.polarity_scores("I am so happy")


# In[26]:


sia.polarity_scores("I am really unset and it is a bad behaviour")


# In[27]:


sia.polarity_scores("you cannot do it, this is worst")


# In[28]:


sia.polarity_scores(examp)


# In[29]:


# Run the polarity score on the entire dataset
res = {}

for index, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    my_id = row['Id']
    res[my_id] = sia.polarity_scores(text)



# In[30]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns = {'index':'Id'})
vaders = vaders.merge(df, how = 'left')


# In[31]:


# Sentiment Score and Metadata
vaders.head()


# In[32]:


#PLot vader result

ax = sns.barplot(data = vaders , x='Score', y='compound')
ax.set_title("Amazon review by compound Score")
plt.show()


# In[33]:


#Positive, Negative and Neutral score for each vader

fig, axs  =plt.subplots(1,3 , figsize=(15,5))
sns.barplot(data =vaders , x= 'Score' ,y= 'pos',ax =  axs[0])
sns.barplot(data =vaders , x= 'Score' ,y= 'neu', ax = axs[1])
sns.barplot(data =vaders , x= 'Score' ,y= 'neg', ax = axs[2])
axs[0].set_title("Positive")
axs[1].set_title("Neutral")
axs[2].set_title("negative")
plt.show()


# # Roberta pretrained model

# In[34]:


#Use a model  trained  of a large corpus data
# Transformer model accounts for the words but also the context related to other words.


# In[35]:


import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[36]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[37]:


#VADERS results on examp
print(examp)
sia.polarity_scores(examp)


# In[38]:


#Run for  Roberta Model
encoded_text = tokenizer(examp,return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# In[39]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


# In[49]:


res = {}

for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        my_id = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key,value in vader_result.items():
                vader_result_rename[f"Vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[my_id] = both
        
    except RuntimeError:
        print(f'Broke for id {my_id}')
    


# In[52]:


result_df = pd.DataFrame(res).T
result_df = result_df.reset_index().rename(columns = {'index':'Id'})
result_df = result_df.merge(df, how = 'left')


# In[53]:


result_df.head()


# In[54]:


result_df.columns


# In[57]:


## Compare scores  between models

sns.pairplot(data  =result_df ,  vars= ['Vader_neg', 'Vader_neu', 'Vader_pos', 
                                        'roberta_neg', 'roberta_neu', 'roberta_pos'],
                                        hue = 'Score',
                                        palette ='tab10')
plt.show()


# In[ ]:


## Review Examples:
## positive 1- Star , Negative 5-Star reviews:


# In[63]:


result_df.query('Score == 1').sort_values('roberta_pos',ascending = False)['Text'].values[0]


# In[65]:


result_df.query('Score == 1').sort_values('Vader_pos',ascending = False)['Text'].values[0]


# In[66]:


# Negative sentime 5-star review

result_df.query('Score == 5').sort_values('roberta_neg',ascending = False)['Text'].values[0]


# In[67]:


result_df.query('Score == 5').sort_values('Vader_neg',ascending = False)['Text'].values[0]


# In[71]:


############## Transformers Pipeline ###########

from transformers import pipeline

sent_pipeline = pipeline('sentiment-analysis')


# In[72]:


sent_pipeline("It is a good day")


# In[73]:


sent_pipeline("Do not speak with cunning people")


# # The End
