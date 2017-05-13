import os
import json
from nltk import word_tokenize, sent_tokenize
from vader import SentimentIntensityAnalyzer
import plotly.plotly as py
import plotly.graph_objs as go

def get_words(sentence):
	return word_tokenize(sentence)


def get_sentences(text):
	return sent_tokenize(text)


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

files = ['Canon G3.txt', 'Apex AD2600 Progressive-scan DVD player.txt', 'Creative Labs Nomad Jukebox Zen Xtra 40GB.txt', 'Nikon coolpix 4300.txt', 'Nokia 6610.txt']

def read_file_1(i):
	path = '/home/jeevan/Desktop/customer review data/'
	data = open(path + str(files[i])).read()
	data1 = data.split('\n')
	data2 = [asdf.split('##',1)[-1] for asdf in data1]
	data3 = ''
	for asdf in data2:
		data3 = data3 + asdf
	data4 = data3.split('[t]')
	return data4


def main():
	# read input text from json file
	# text = read_file(0)
	# tokenize the paragraph
	# para_tokens = get_words(text)
	# get sentences from the paragraph
	# sents = get_sentences(text)
	for i in range(len(files)):
		sents = read_file_1(i)
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
	    	title=str(files[i]),
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
		py.plot(fig, filename=files[i])

# def plotMatrix3DByMatPlotLib(matScores,m,n):
#     nx, ny = m, n
#     x = range(nx)
#     y = range(ny)    
#     hf = plt.figure()
#     ha = hf.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
#     ha.plot_surface(X, Y, matScores[0:m,0:n])    
#     plt.show()


if __name__ == "__main__":
	main()