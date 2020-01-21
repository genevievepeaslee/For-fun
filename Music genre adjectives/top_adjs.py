import sqlite3
from collections import defaultdict
import spacy
import re
import gensim
from spacy.tokenizer import Tokenizer
from nltk.stem import WordNetLemmatizer

spacy_proc = spacy.load('en', disable=['parser', 'textcat', 'ner'])
infix_re = re.compile(r'''[~]''')  # for not splitting on intraword hyphens
new_tokenizer = Tokenizer(spacy_proc.vocab, infix_finditer=infix_re.finditer) 
spacy_proc.tokenizer = new_tokenizer
spacy_proc.lemmatizer = WordNetLemmatizer  #  more coverage than Spacy's built-in lemmatizer

# TEXT CLEANING
PUNCT = '/!?.,@#$%^&*()_<>[]{};:'
punct_trans = str.maketrans(dict.fromkeys(PUNCT, " "))
quote_trans = str.maketrans(dict.fromkeys('''’'“"”''', ""))

def simple_clean(doc: str) -> str:
    """
    Remove single and double quotes; replace other punctuation (not hyphens) with a space
    @return: string
    """
    # remove single and double quotes    
    doc = doc.translate(quote_trans)
    # replace punct with space
    return doc.translate(punct_trans)
    
def preprocess(doc: str) -> list:
    """
    NOTE: lemmatizer seems to favor high precision, doesn't touch a lot of derivations. Still better than nothing.
    @param doc: 
    @return: list of strings
    """
    doc = simple_clean(doc)
    doc = spacy_proc(doc)
    proc_doc = []
    for i in range(len(doc)):
        if doc[i].pos_ == 'ADJ':
            token = doc[i].text
            # Skip proper names mislabeled as adj as well as adjectives like "Italian"; skip tokens containing numbers
            if token.islower() and token.isalpha():  
                proc_doc.append(doc[i].lemma_)
    return proc_doc

# GET REVIEW CONTENT
con = sqlite3.connect('/kaggle/input/pitchfork-data/database.sqlite')
cur = con.cursor()

# To use gensim's tf-idf calculator we need a dict with genres as keys and all the reviews within 
# that genre concatenated into one string to be treated as a 'document'.

NUM_REVIEWS = 18393
content_by_genre = defaultdict(list)
i = 0
for row in cur.execute('''SELECT c.content AS content, g.genre AS genre 
                          FROM content AS c 
                              INNER JOIN genres AS g ON c.reviewid = g.reviewid'''):
    if i == NUM_REVIEWS:
        break
    else:
        processed_review = preprocess(row[0])
        genre = row[1]
        content_by_genre[genre].extend(processed_review)
        i += 1
genres = [key for key in content_by_genre.keys()]
con.close()

# CALCULATE SCORES
# Find highly representative words for each genre by computing tf-idf scores 
word_ids = gensim.corpora.Dictionary([doc for doc in content_by_genre.values()])
counts = [word_ids.doc2bow(doc) for doc in content_by_genre.values()]
tfidf_model = gensim.models.tfidfmodel.TfidfModel(counts)
corpus_tfidf = tfidf_model[counts]  
# Note: corpus_tfidf[i] is a list of tuples, one for each token present in genre[i], of the form (<word_id>, <tfidf_score>)

# PRINT
# Print highest-scoring adjectives for each genre
TOP_NUM = 8
for i in range(len(genres)):
    print(genres[i])
    sorted_scores = sorted(corpus_tfidf[i], key=lambda x: x[1], reverse=True)
    top = [word_ids[x[0]] for x in sorted_scores[:TOP_NUM]]
    print(top)
