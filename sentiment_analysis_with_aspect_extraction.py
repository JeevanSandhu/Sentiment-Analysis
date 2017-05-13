##################################################
##### Sentiment Analysis after Aspect Extraction
##################################################

from __future__ import division
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk import pos_tag
import os
import json
from operator import itemgetter, attrgetter, methodcaller
from vader import SentimentIntensityAnalyzer

def get_sentences(text):
	return sent_tokenize(text)

def get_words(sentence):
	return word_tokenize(sentence)

def tag_pos(tokens):
	return pos_tag(tokens)

def stopword_rem(wordlist):
	stopword_list = set(stopwords.words('english'))
	return [stopped for stopped in wordlist if stopped not in stopword_list]

def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    import itertools, nltk, string    
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent)) for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower() for key, group in itertools.groupby(all_chunks, lambda (word,pos,chunk): chunk != 'O') if key]
    return [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    import itertools, nltk, string
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words if tag in good_tags and word.lower() not in stop_words and not all(char in punct for char in word)]
    return candidates

def score_keyphrases_by_textrank(text, n_keywords=0.05):
    from itertools import takewhile, tee, zip
    import networkx, nltk
    # tokenize for all words, and extract *candidate* words
    words = [word.lower()
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    candidates = extract_candidate_words(text)
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)
    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1] for word_rank in sorted(ranks.iteritems(), 
    	key=lambda x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)
    return sorted(keyphrases.iteritems(), key=lambda x: x[1], reverse=True)

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
	path = '/home/jeevan/Desktop/AmazonReviews/laptops/'
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

def main():
	#read input text from json file
	text = read_file(0)

	#tokenize the paragraph
	para_tokens = get_words(text)

	#get sentences from the paragraph
	sents = get_sentences(text)

	#tokenize each sentence
	sent_tokens = [get_words(tokens) for tokens in sents]
	
	#remove stopwords from each sentence
	stopped_sent = [stopword_rem(sentence) for sentence in sent_tokens]

	#Extract aspects from the sample text
	text_score = score_keyphrases_by_textrank(text)

	#Calculate the frequency of the aspects
	freq = []
	for words, score in text_score:
		for w in words.split():
			freq.append(w)

	#Calculate the frequency of each word
	freq_uniq = get_frequency(freq, freq)

	#
	aspect_list = ['laptop', 'keyboard', 'keys', 'screen', 'graphics', 'processor', 'display', 'body', 'size', 'mouse', 'trackpad', 'track', 'battery', 'sensors']

	#get top 10 aspects based on frequency
	scores = []
	words = []
	for word1 in aspect_list:
		for word2 in freq_uniq.keys():
			if word1 == word2:
				words.append(word1)
				scores.append(freq_uniq[word1])

	top10 = zip(words, scores)
	top10 = sorted(top10, key=itemgetter(1))
	top10.reverse()

	#get the lines that the aspects occur in
	i=0
	aspect_sent = []
	aspect_topic = []
	for qwer in sents:
		for top in top10:
			for bla in qwer.split():
				if top[0] == bla:
					aspect_sent.append(i)
					aspect_topic.append(top[0])
		i=i+1

	# aspect_sent = zip(aspect_topic,aspect_sent)

	# aspect1 = ['graphics', 'screen', 'size']
	# aspect2 = ['keyboard', 'keys', 'key']

	aspect_sent_uniq = list(set(aspect_sent))

	sid = SentimentIntensityAnalyzer()

	sentiment_scores = [sid.polarity_scores(sents[i]) for i in aspect_sent_uniq]

	pos_sents = ""
	neg_sents = ""

	print "\n\n Positive: \n"
	for i in range(0,len(aspect_sent_uniq)):
		if sentiment_scores[i]['pos'] > sentiment_scores[i]['neg']:
			print sents[aspect_sent_uniq[i]]
			pos_sents = sents[aspect_sent_uniq[i]] + " "

	print "\n\nPositive Sentences Polarity:"
	print sid.polarity_scores(pos_sents)

	print "\n\n Negative: \n"
	for i in range(0,len(aspect_sent_uniq)):
		if sentiment_scores[i]['neg'] > sentiment_scores[i]['pos']:
			print sents[aspect_sent_uniq[i]]
			neg_sents = sents[aspect_sent_uniq[i]] + " "

	print "\n\nNegitive Sentences Polarity:"
	print sid.polarity_scores(neg_sents)