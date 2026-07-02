import pandas as pd
import emoji
import re

from sklearn.model_selection import train_test_split

# ===== config =====
RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10 

emoticon_pattern = re.compile(
    r'(?<!\d)(?:(?:[:;=xX])(?:-)?(?:\)|\(|D|P|p|/|\\|\||\*|O|o|>|<|3|\]|\[|;|C|c)+)'
)

def remove_all_emojis(text):
    if not isinstance(text, str):
        return text
    # remove unicode emojis
    text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)
    # remove ASCII emoticons
    text = emoticon_pattern.sub('', text)
    return text

def demojis(text):
    # if not isinstance(text, str):
    #     return text
    text = emoji.demojize(text)
    return text

def keep_hashtag_words(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'#(\w+)', r'\1', text)

df = pd.read_csv('bachelor_dataset.csv')

# df['clean_text'] = df['text'].apply(remove_all_emojis)
df['clean_text'] = df['text'].apply(emoji.demojize)
df['clean_text'] = df['clean_text'].str.replace(r'(@\w+\s*)+', '', regex=True) #remove usernames
df['clean_text'] = df['clean_text'].apply(keep_hashtag_words)

df_trainval, df_test = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["label"],
    random_state=RANDOM_STATE,
)

val_size_relative = VAL_SIZE / (1.0 - TEST_SIZE)  # 0.10 / 0.90 = 0.111...
df_train, df_val = train_test_split(
    df_trainval,
    test_size=val_size_relative,
    stratify=df_trainval["label"],
    random_state=RANDOM_STATE,
)

def show_split_stats(name, frame):
    print(f"\n{name} size: {len(frame)}")
    print(frame["label"].value_counts().sort_index())

print("\nFinal split sizes")
print(f"Train: {len(df_train)}  Validation: {len(df_val)}  Test: {len(df_test)}")
show_split_stats("Train", df_train)
show_split_stats("Validation", df_val)
show_split_stats("Test", df_test)

df_train = df_train.drop(['tweet_id', 'text' ,'queried_user'], axis=1)
df_val = df_val.drop(['tweet_id', 'text' ,'queried_user'], axis=1)
df_test = df_test.drop(['tweet_id', 'text' ,'queried_user'], axis=1)

df_train.to_csv("train.csv", index=False)
df_val.to_csv("val.csv", index=False)
df_test.to_csv("test.csv", index=False)

