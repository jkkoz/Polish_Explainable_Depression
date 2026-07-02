import pandas as pd

df = pd.read_csv('retrieved.csv')

rel_cols = df[['tweet_id','queried_user','text', 
                      'created_at', 'quotes', 'replies', 'retweets']].copy()

no_rt = rel_cols[~rel_cols['text'].str.startswith('RT ', na=False)]

no_rt.to_csv('depressed_cleaned.csv', index=False)
