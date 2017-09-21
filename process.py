import config
import json
import datetime
import re
import numpy as np
from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# reading data to mongodb source collection
# mongoimport --db Fundamentals --collection source --drop --type json --file "[FilePath to tweetsfile]"


# writing data to mongodb processed collection
def load_processed_collection(n = 0):
    client = MongoClient(config.mongoConnectionString)
    database = client[config.mongoDatabase]
    source_collection = database['source']
    processed_collection = database['processed']
    error_collection = database['error']

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    def clean(doc):
        punc_free = ''.join(ch for ch in doc.lower() if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    def clean_stopwords(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    if n > 0:
        tweets = source_collection.find(limit = n)
    else:
        tweets = source_collection.find()

    i = 0
    for tweet in tweets:
        try:
            tweetid = tweet['id']
            userid = tweet['user']['id']

            place = tweet['place']
            if not place or place['country_code'] != 'US':  # only interested in us tweets
                continue

            state = place['full_name'][-2:]  # get last two characters to get us state

            lat = None
            lon = None
            if place['place_type'] == 'city':
                coordinates = place['bounding_box']['coordinates']
                np_coordinates = np.array(coordinates[0])
                lat = round(np.mean(np_coordinates[:, 1]), 5)
                lon = round(np.mean(np_coordinates[:, 0]), 5)

            dtime = datetime.datetime.fromtimestamp(int(tweet['timestamp_ms'][0:-3])).strftime(
                '%Y-%m-%d %H:%M')

            text = tweet['text']
            text = re.sub(r"[\n\r]", " ", text)  # remove newlines
            text = re.sub(r"http\S+", "", text)  # remove hyperlinks
            text = re.sub(r"@\S+", "", text)  # remove @users

            cleantext = clean(text)
            cleantext = re.sub(r"[^A-Za-z0-9\s]", "", cleantext)
            cleantext = cleantext.strip()

            nostops = clean_stopwords(text)
            nostops = re.sub(r"[^A-Za-z0-9\s]", "", nostops)
            nostops = nostops.strip()

            # When there is no text left after cleaning, continue
            if not cleantext:
                continue

            processed_collection.insert_one({
                'id': tweetid,
                'userid': userid,
                'state': state,
                'lat': lat,
                'lon': lon,
                'datetime': dtime,
                'text': text,
                'cleantext': cleantext,
                'nostops': nostops
            })

        except:
            error_collection.insert_one(tweet)

        i += 1
        if i % 10000 == 0:
            print("Processed {} tweets".format(i))


# method to perform sentiment analysis
def perform_sentimentanalysis(n = 0):
    client = MongoClient(config.mongoConnectionString)
    database = client[config.mongoDatabase]
    processed_collection = database['processed']

    sid = SentimentIntensityAnalyzer()

    if n > 0:
        tweets = processed_collection.find(limit = n)
    else:
        tweets = processed_collection.find()

    i = 0
    for tweet in tweets:
        scores = sid.polarity_scores(tweet['cleantext'])

        tweet["compound"] = scores['compound']
        tweet["neg"] = scores['neg']
        tweet["neu"] = scores['neu']
        tweet["pos"] = scores['pos']
        processed_collection.save(tweet)

        i += 1
        if i % 10000 == 0:
            print("Processed {} tweets".format(i))

# method to perform lda analysis

