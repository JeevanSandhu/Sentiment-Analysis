##################################################
##### Sentiment Analysis with 3D Visualization
##################################################

import os
import json
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from vader import SentimentIntensityAnalyzer
import plotly.plotly as py
import plotly.graph_objs as go


def get_sentences(text):
	return sent_tokenize(text)


def get_words(sentence):
	return word_tokenize(sentence)


def tag_pos(tokens):
	return pos_tag(tokens)


def stopword_rem(wordlist):
	stopword_list = set(stopwords.words('english'))
	return [stopped for stopped in wordlist if stopped not in stopword_list]


def get_frequency(wordlist, paralist):
	from collections import Counter
	count = Counter(paralist)
	new_count = []
	for word in wordlist:
		new_count.append(count[word])
	frequency = dict(zip(wordlist,new_count))
	return frequency


def read_file(file_number):
	#Open customer review files and read Review Titles and Content from them
	path = 'Dataset/AmazonReviews/laptops/'
	filenames = []
	for filename in os.listdir(path):
		filenames.append(filename)
	products = []
	reviewTitle = []
	reviewContent = []
	with open(path + filenames[file_number]) as dataFile:
		data = json.load(dataFile)
		products.append(data['ProductInfo'])
		for reviews in data['Reviews']:
			reviewTitle.append(reviews['Title'])
			reviewContent.append(reviews['Content'])
	sample = ''
	for rev in reviewContent:
		sample = sample + ' ' + rev
	return sample


def read_file_1():
	path = 'Dataset/customer_review_data/'
	data = open(path + 'Canon G3.txt').read()
	data1 = data.split('\n')
	data2 = [asdf.split('##',1)[-1] for asdf in data1]
	data3 = ''
	for asdf in data2:
		data3 = data3 + asdf
	data4 = data3.split('[t]')
	return data4


def main():
	text = read_file_1()
	review_tokens = [get_words(asdf) for asdf in text]
	stopped_sent = [stopword_rem(sentence) for sentence in review_tokens]
	text_score = [score_keyphrases_by_textrank(asdf) for asdf in text]

	sents = []
	for i in stopped_sent:
		asdf = ''
		for j in i:
			asdf = asdf + j + ' '
		sents.append(asdf)


	sid = SentimentIntensityAnalyzer()
	sentiment_scores = [sid.polarity_scores(sent) for sent in sents]
	bla = [[asdf['pos'],-1*asdf['neg']] for asdf in sentiment_scores]
	bla.sort()
	qwer = range(len(bla))
	data = [
		go.Surface(
			z=bla,
        	x=qwer,
        	y=qwer
        )
    ]
	layout = go.Layout(
    	title='Canon G3',
    	autosize=False,
    	width=500,
    	height=500,
    	margin=dict(
	        l=65,
        	r=50,
        	b=65,
        	t=90
    	)
    )
	fig = go.Figure(data=data, layout=layout)
	py.plot(fig, filename='Sentiment Analysis')