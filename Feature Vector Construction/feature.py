import pandas as pd
import re
import morfeusz2
import spacy
from tqdm import tqdm
from functools import lru_cache

FIRST_PERSON: frozenset[str] = frozenset({
    "ja", "mnie", "mię", "mi", "mną", 
    "my", "nas", "nam","nami",
    "mój", "moja", "moje", 
    "mojego", "mego", "mojej", "mej", 
    "mojemu", "memu", 
    "moim", "mym","moją", "mą","me", 
    "moich", "moimi", "moi", "mych",
    "nasz", "nasza", "nasze", 
    "naszego", "naszej", 
    "naszemu",
    "naszym", "naszą",
    "naszych", "naszymi", "nasi",
    # "bym", "byśmy"
})

RELEVANT = re.compile(r"^[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{2,}$")

df = pd.read_csv('corrected_dataset.csv')

m = morfeusz2.Morfeusz()
nlp = spacy.load('pl_core_news_lg')

@lru_cache(maxsize=100_000)
def classify_cached(token):
    return classify(token)

def classify(token):
    analysis = m.analyse(token)
    print(analysis)

    is_verb = False
    is_pron = False
    is_sing = False
    is_past = False
    temp_cond = False
    is_cond = False 

    # is the token itself a first person pronoun?
    if token in FIRST_PERSON:
        is_pron = True
        return is_verb, is_pron, is_sing, is_past, is_cond

    for analysed in analysis: #for each part of the resulting morphological list

        tag = analysed[2][2] #get tag
        lemma = analysed[2][1] #get lemma (base form of the word)
        tags = tag.split(":")  
        pos = tags[0]   #get the part of speech

        # sanity check 
        if lemma in FIRST_PERSON:
            is_pron = True

        # verb is conditional; temp variable because conditionals are split over two tags    
        if pos in ("fin", "bedzie") and "pri" in tags:
            if "sg" in tags:
                is_verb = True
                is_sing = True
            elif "pl" in tags:
                is_verb = True
    
        # verb is conditional    
        if pos == "part" and lemma in ("by:T", "by:M"):
            temp_cond = True 

        # verb in 1st person past or conditional
        if pos == "aglt" and "pri" in tags:
            if not temp_cond: # is in past tense
                if "sg" in tags:
                    is_verb = True
                    is_past = True
                    is_sing = True
                elif "pl" in tags:
                    is_verb = True
                    is_past = True
            else: # is conditional
                if "sg" in tags:
                    is_verb = True
                    is_sing = True
                    is_cond = True
                elif "pl" in tags:
                    is_verb = True
                    is_cond = True

    return is_verb, is_pron, is_sing, is_past, is_cond

def extract(tweet) -> dict[str, float]:

    if not isinstance(tweet, str) or tweet.strip() == "":
        return {
            "n_1st_verb": 0,
            "n_1st_pronoun": 0,
            "n_sing": 0,
            "n_words": 0,
            "verb_count": 0,
            "pron_count": 0,
            "rate_sing": 0
        }

    words = tweet.split()

    nverb = 0
    npron = 0
    nsing = 0
    npast = 0
    ncond = 0
    nwords = max(len(words),1)

    for word in words:
        if not RELEVANT.match(word): # filter out invalid words; speeds up the process
            continue
        
        verb, pron, sing, past, cond = classify_cached(word)
        nverb += int(verb)
        npron += int(pron)
        nsing += int(sing)
        npast += int(past)
        ncond += int(cond)

    return {
            "n_1st_verb": nverb,
            "n_1st_pronoun": npron,
            "n_sing": nsing,
            "n_past": npast,
            "n_cond" :ncond,
            "n_words": nwords,
            "rate_verb": nverb / nwords,        # number of verbs in 1st person over the total number of words
            "rate_pron": npron /nwords,         # number of 1st person pronouns over the total number of words
            "rate_sing": nsing / max(nverb,1),  # number of 1st person verbs in singular form over all 1st person verbs
            "rate_past": npast / max(nverb,1),  # number of 1st person verbs in past tense over all 1st person verbs
            "rate_cond": ncond / max(nverb,1)   # number of 1st person verbs in conditional form over all 1st person verbs
        }

tqdm.pandas()

features = df['corrected_text'].progress_apply(extract).apply(pd.Series)
df = pd.concat([df, features], axis=1)
df.to_csv('with_features.csv', index=False)

"""

Consider extract("byłbym bylibyśmy byłby był byłem"). This will return 
{
'n_1st_verb': 3, 
'n_1st_pronoun': 0, 
'n_sing': 2, 
'n_past': 1, 
'n_cond': 2, 
'n_words': 5, 
'rate_verb': 0.6, 
'rate_pron': 0.0, 
'rate_sing': 0.6666666666666666, 
'rate_past': 0.3333333333333333, 
'rate_cond': 0.6666666666666666
}

We have three 1s person verbs: byłbym, bylibyśmy, byłem. 
Zero 1st person pronouns.
Two of the verbs are in singular form: byłbym, byłem.
One is in past tense: byłem.
Two are conditional: byłbym, bylibyśmy.
The string contains 5 words in total.

Thus, rate_verb is three 1st person verbs over all words: 3/5 = 0.6
rate_pron is zero 1st person pronouns over all words: 0/5 = 0.0
rate_sing is two 1st person verbs in singular form over all 1st person verbs: 2/3 = 0.666.
rate_past is one 1st person verbs in past tense over all 1st person verbs: 1/3 = 0.666.
rate_cond is two 1st person conditional verbs over all 1st person verbs: 2/3 = 0.666.

"""
