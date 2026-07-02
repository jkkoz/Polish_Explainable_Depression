import pandas as pd
import emoji
import re
from transformers import pipeline
import torch 
from tqdm import tqdm
import morfeusz2
import spacy
from functools import lru_cache

torch.cuda.empty_cache()

emoticon_pattern = re.compile(
    r'(?<!\d)(?:(?:[:;=xX])(?:-)?(?:\)|\(|D|P|p|/|\\|\||\*|O|o|>|<|3|\]|\[|;|C|c)+)'
)


df = pd.read_csv('bachelor_dataset.csv')

def remove_all_emojis(text):
    if not isinstance(text, str):
        return text
    # remove unicode emojis
    text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)
    # remove ASCII emoticons
    text = emoticon_pattern.sub('', text)
    return text

def demojis(text):
    if not isinstance(text, str):
        return text
    text = emoji.demojize(text)
    return text

def keep_hashtag_words(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'#(\w+)', r'\1', text)

def has_emoji(text):
    has_unicode_emoji = any(c in text for c in emoji.EMOJI_DATA)
    has_emoticon = emoticon_pattern.search(text) is not None
    if has_unicode_emoji or has_emoticon:
        return 1
    else: 
        return 0
    
def has_absolutist(text): 
    absolutist_words = [
        "absolutnie", "wszyscy", "wszystkie", "zawsze",
        "komplenty", "kompletna", "całkowity", "całkowita",
        "kompletnie", "całkowicie", "stały", "stała", "stale", "ciągle",
        "zdecydowanie", "na pewno", "definitywnie", "cały", "kiedykolwiek",
        "każdy", "każda", "wszystko",
        "cały", "cała", "pełny", "pełen", "pełna", "musi", "muszę", "muszą",
        "nigdy", "nic", "zupełnie", "totalnie", "całość", "wcale"
    ]
    
    if any(word in text for word in absolutist_words): 
        return 1 
    else: 
        return 0

df['emoji'] = df['text'].apply(has_emoji)
df['absolutist'] = df['text'].astype(str).apply(has_absolutist) 

df.to_csv('emo_abs_dataset.csv', index=False)
