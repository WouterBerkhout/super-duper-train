import csv
import string
import time

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk.stem.wordnet import WordNetLemmatizer


_stopwords_en = set(stopwords.words('english'))
_punctuations = set(string.punctuation)
_unwanted_entities = _stopwords_en | _punctuations

# This is the function makeing the lemmatization
_lemma = WordNetLemmatizer()
_sid = SentimentIntensityAnalyzer()

def clean(text):
    unwanted_free = " ".join([i for i in text.lower().split() if i not in _unwanted_entities])
    normalized = " ".join(_lemma.lemmatize(word) for word in unwanted_free.split())
    return normalized

def get_sentiment_scores(text):
    pass

def extend_csv_with_sentiment(inputfile, outputfile, delimiter=';', text_row='text'):
    csv_in = open(inputfile, 'r', encoding='utf-8')
    csv_out = open(outputfile, 'w', encoding='utf-8')

    fieldnames = [ 'dateime', 'state', 'latlon', 'text', 'pos', 'neg', 'neu', 'compound' ]
    writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
    writer.writeheader()
    reader = csv.DictReader(csv_in, delimiter=delimiter)

    for row in reader:
        scores = _sid.polarity_scores(clean(row[text_row]))
        writer.writerow({**row, **scores})

    csv_in.close()
    csv_out.close()
