import json
import datetime
import re
import numpy as np
import sys
sys.setdefaultencoding('utf8')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string



def find_tweet(tweet_id, inputfile, outputfile):
	with open(outputfile, "w") as output:
		with open(inputfile, "r") as input:
			for line in input:
				tweet = json.loads(line)
				if tweet['id'] == tweet_id:
					output.write(line)
					print("tweet found and written to outputfile")
					return
			print("tweet not found.")


def save_tweets(inputfile, outputfile, n):
	i = 0
	with open(outputfile, "w") as w:
		with open(inputfile, "r") as f:
			for line in f:
				if n > 0 and i > n:
					break

				w.writelines([line])
				i += 1



def save_csv(inputfile, outputfile, errorfile, n = 0, debug = False):
	i = 0

	stop = set(stopwords.words('english'))
	exclude = set(string.punctuation)
	lemma = WordNetLemmatizer()

	def clean(doc):
		stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
		punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
		normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
		return normalized

	with open(errorfile, "w") as err:
		with open(outputfile, "w") as w:
			w.write("id;dateime;state;latlon;text\n")
			with open(inputfile, "r") as f:
				for line in f:
					if n > 0 and i > n:
						break

					try:
						tweet = json.loads(line)
						tweetid = tweet['id']

						place = tweet['place']
						if not place or place['country_code'] != 'US': # only interested in us tweets
							continue

						state = place['full_name'][-2:] # get last two characters to get us state
						latlon = ''
						if place['place_type'] == 'city':
							coordinates = place['bounding_box']['coordinates']
							np_coordinates = np.array(coordinates[0])
							lat = round(np.mean(np_coordinates[:,1]), 5)
							lon = round(np.mean(np_coordinates[:,0]), 5)
							latlon = "{},{}".format(lat, lon)

						dtime = datetime.datetime.fromtimestamp(int(tweet['timestamp_ms'][0:-3])).strftime('%Y-%m-%d %H:%M')

						text = tweet['text']
						text = re.sub(r"\n|\r", " ", text) # remove newlines
						text = re.sub(r"http\S+", "", text) # remove hyperlinks
						text = re.sub(r"@\S+", "", text) # remove @users
						text = clean(text) # remove common words and lemmatize
						text = re.sub(r"[^AZa-z0-9\s]", "", text) # remove weird characters
						text = text.strip()

						# When there is no text left after cleaning, continue
						if not text:
							continue

						# We can do text analysis here

						if debug:
							csvline = "{};{};{};{};".format(tweetid, dtime, state, latlon) + text + "\n"
						else:
							csvline = "{};{};{};".format(dtime, state, latlon) + text + "\n"

						w.write(csvline)

					except:
						lineNo = i + 1
						print("Error when parsing line " + str(lineNo))
						err.write(line)

					i += 1

					if i % 10000 == 0:
						print("Processed {} tweets".format(i))
