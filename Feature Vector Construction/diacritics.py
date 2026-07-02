import pandas as pd
import re
import morfeusz2
from wordfreq import word_frequency
from functools import lru_cache
from tqdm import tqdm

# mapping of letters to their possible diacritic variation
DIACRITIC_MAP : dict[str, list[str]] = {
    "a": ["ą"],
    "c": ["ć"],
    "e": ["ę"],
    "l": ["ł"],
    "n": ["ń"],
    "o": ["ó"],
    "s": ["ś"],
    "z": ["ź", "ż"],
}
 
# immutable set of Polish diacritics
POLISH_DIACRITICS : frozenset[str] = frozenset("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

# initialize Morfeusz2 instance
morf = morfeusz2.Morfeusz()

# returns true if the text contains a diacritic; returns false otherwise
def has_diacritics(text):
    return any(c in POLISH_DIACRITICS for c in text)

# gets rid of punctuation 
def strip_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# returns true if Morfeusz2 recognises the word with added diacritics
# "ign" (ignotum) is the tag that represents that the segment is unknown to the analyzer
def is_valid(token):
    results = morf.analyse(token)
    return bool(results) and results[0][2][2] != "ign"

@lru_cache(maxsize=65536)
def restore_token_cached(token, tweet_diac):
    return restore_token(token, tweet_diac)

def generate_words(token):
    variants = {token}
    for i, c in enumerate(token):
        if c in DIACRITIC_MAP:
            for d in DIACRITIC_MAP[c]:
                variants.add(token[:i] + d + token[i+1:])
    return variants

def restore_token(token, tweet_diac):
    lower = token.lower()
    token_diac = has_diacritics(lower)
    
    if not token.isalpha() or len(token) <= 2:
        return token
    
    if tweet_diac and token_diac:
        return token
    
    candidates = generate_words(lower)
    # print(candidates)
    valid = [word for word in candidates if is_valid(word)]

    # print(valid)

    if not valid: 
        return token
    
    best = max(valid, key=lambda c: word_frequency(c, 'pl'))

    if lower in valid:
        bare_freq = word_frequency(lower, 'pl')
        best_freq = word_frequency(best, 'pl')
        if best == lower or best_freq < bare_freq:
            return token
          
    return best

def restore(text):
    tweet_diac = has_diacritics(text)
    clean = strip_punctuation(text)
    tokens = clean.split()
    return " ".join(restore_token_cached(tok, tweet_diac) for tok in tokens)

df = pd.read_csv('bachelor_dataset.csv')
# # tweet = strip_punctuation(df['text'].iloc[13])
# # tokens = tweet.split()
# # print(tokens[4], restore_token(tokens[4], has_diacritics(tweet)))

# print(restore("w szczebrzeszynie chrzaszcz brzmi w trzcinie"))
tqdm.pandas()
df['corrected_text'] = df['text'].astype(str).progress_apply(restore)
df = df.drop(['tweet_id', 'queried_user', 'text'],axis=1)
df.to_csv('corrected_dataset.csv', index=False)
