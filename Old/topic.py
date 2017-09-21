import csv
import string
import time

from gensim import corpora
from gensim.models import LdaMulticore
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


_stopwords_en = set(stopwords.words('english'))
_punctuations = set(string.punctuation)
_unwanted_entities = _stopwords_en | _punctuations

# This is the function makeing the lemmatization
_lemma = WordNetLemmatizer()

# In this function we perform the entire cleaning
def clean(doc):
    unwanted_free = " ".join([i for i in doc.lower().split() if i not in _unwanted_entities])
    normalized = " ".join(_lemma.lemmatize(word) for word in unwanted_free.split())
    return normalized

def train_model(docs):
    cleaned_docs = [clean(doc).split() for doc in docs]
    dictionary = corpora.Dictionary(cleaned_docs)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned_docs]
    Lda = LdaMulticore
    ldamodel = Lda(doc_term_matrix, num_topics=100, id2word = dictionary, passes=10, workers=3)

def train_model_from_csv(filename, delimiter=';'):
    dictionary = corpora.Dictionary()
    with open(filename, 'rb') as f:
        # Create a csv DictReader for easy access of the text fields
        reader = csv.DictReader(f, delimiter=delimiter)

        print('Starting creation of dictionary')
        t_start = time.time()
        for row in reader:
            dictionary.add_documents([clean(row['text']).split()])

        t_end = time.time()
        print('Dictionary created')
        print('Took {} sec to create dictionary'.format(t_end-t_start))
        print('length of dictionary')
        print(len(dictionary.keys()))

        # Reset line pointer for the file so we can loop again.
        # This time we create the doc_term_matrix
        f.seek(0)

        print('Starting creation of doc_term_matrix')
        t_start = time.time()
        doc_term_matrix = [dictionary.doc2bow(row['text'].split()) for row in reader]
        t_end = time.time()
        print('doc_term_matrix created')
        print('Took {} sec to create doc_term_matrix'.format(t_end-t_start))
        print('length of doc_term_matrix')
        print(len(doc_term_matrix))

        print('Start traing of model')
        t_start = time.time()
        Lda = LdaMulticore
        ldamodel = Lda(doc_term_matrix, num_topics=100, id2word = dictionary, passes=1, workers=3)
        t_end = time.time()
        print('Model trained')
        print('Took {} sec to train the model'.format(t_end-t_start))
        ldamodel.save('data/lda_model.gensim')
        return ldamodel
