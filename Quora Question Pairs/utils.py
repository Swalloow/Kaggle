from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from multiprocessing import Pool
import numpy as np
import pandas as pd
import re


# Local variables
num_cores = 4


# Split dataframe into number of cores (4)
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# 10k -> 10000
def substitute_thousands(text):
    matches = re.finditer(r'[0-9]+(?P<thousands>\s{0,2}k\b)', text, flags=re.I)
    result = ''
    len_offset = 0
    for match in matches:
        result += '{}000'.format(text[len(result)-len_offset:match.start('thousands')])
        len_offset += 3 - (match.end('thousands') - match.start('thousands'))
    result += text[len(result)-len_offset:]
    return result


# Which -> Which/JJ
def eng_pos_tagger(text):
    # Using averaged_perceptron_tagger, maxent_treebank_pos_tagger
    # nltk.download('averaged_perceptron_tagger)
    tagger = ["/".join(i) for i in pos_tag(text.split())]
    return ' '.join(tagger)


# remove stop words, stemming words
def text_cleaning(text, remove_stop_words=True, stem_words=True):
    # Clean the text with the option to remove stop_words and to stem words.
    # Clean the text
    text = text.lower()
    text = substitute_thousands(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r"the us", "America", text)
    text = re.sub(r" uk ", " england ", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r" j k ", " jk ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stopwords.words('english')]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


# Data preprocessing
def preprocessing_test(df):
    question1 = df['question1'].apply(lambda x: text_cleaning(str(x)))
    question2 = df['question2'].apply(lambda x: text_cleaning(str(x)))
    return pd.DataFrame({'question1': question1, 'question2': question2})


# Make features
def make_features(df):
    print("Preprocessing...")
    train = parallelize_dataframe(df, preprocessing_test)
    
    print("Making word features...")
    df['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
    df['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
    df['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
    df['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
    df['len_word_q2'] = df['len_word_q2'].fillna(0)
    df['avg_world_len1'] = df['len_q1'] / df['len_word_q1']
    df['avg_world_len2'] = df['len_q2'] / df['len_word_q2']
    train['diff_avg_word'] = df['avg_world_len1'] - df['avg_world_len2']
    train['word_match'] = df.apply(word_match_share, axis=1, raw=True)    
    
    print("Making bag of words features...")
    all_questions = train['question1'].append(train['question2'])
    tfidf = TfidfVectorizer(lowercase=True, binary=True).fit(all_questions)
    q1_tfidf1 = tfidf.transform(train['question1'])
    q2_tfidf1 = tfidf.transform(train['question2'])
    tfidf = TfidfVectorizer(lowercase=True, binary=True, ngram_range=(1,3), analyzer='word', \
                            max_features=100000, max_df=0.5, min_df=30, use_idf=True).fit(all_questions)
    q1_tfidf2 = tfidf.transform(train['question1'])
    q2_tfidf2 = tfidf.transform(train['question2'])
    count = CountVectorizer(lowercase=True, binary=True, ngram_range=(1,10), analyzer='char', \
                            max_features=300000, max_df=0.999, min_df=50).fit(all_questions)
    q1_count = count.transform(train['question1'])
    q2_count = count.transform(train['question2'])
    
    print("Calculate distances...")
    train['tf_distance1'] = paired_cosine_distances(q1_tfidf1, q2_tfidf1)
    train['tf_distance2'] = paired_cosine_distances(q1_tfidf2, q2_tfidf2)
    train['cnt_distance'] = paired_cosine_distances(q1_count, q2_count)
    train['jaccard_dist'] = df.apply(lambda x: jaccard_distance(set(str(x.question1).split(' ')), \
                                                                set(str(x.question2).split(' '))), axis=1)
    return train
    