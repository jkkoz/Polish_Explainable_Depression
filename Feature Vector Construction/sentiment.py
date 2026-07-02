from transformers import pipeline
import emoji
import re
import pandas as pd
import torch 
from tqdm import tqdm

torch.cuda.empty_cache()

# quickly clean the text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

df = pd.read_csv('bachelor_dataset.csv')

df['clean_text'] = df['text'].apply(emoji.demojize)
df['clean_text'] = df['clean_text'].apply(preprocess)
df = df.drop(['tweet_id', 'text' ,'queried_user'], axis=1)

nlp = pipeline(
    "sentiment-analysis",
    model="bardsai/twitter-sentiment-pl-base",
    truncation=True,
    max_length=512
)

tqdm.pandas()
df['sentiment'] = df['clean_text'].progress_apply(nlp)
print(df.iloc[4744]) # print example

df.to_csv('with_sentiment.csv', index=False)
