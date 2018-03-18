import argparse
import json
import nltk
import os
import shutil
import string
import zipfile

# TODO : Handle cases where words are separated by punctuations themselves
# Generate unigram and store

# Done both; this file is done
stopwords = nltk.corpus.stopwords
stemmer = nltk.stem.snowball.SnowballStemmer('english')

parser = argparse.ArgumentParser(description = 'Utilities for preprocessing documents')
parser.add_argument('--unzip', type = str, help = 'Unzip the documents file, and remove unnecessary payload files'
	, default = None)
parser.add_argument('--generate_unigram_inverted_index', type = str, help = 'Generate a unigram inverted index\
	from the given document list', default = '20_newsgroups')

def unzip(filename = '20_newsgroups.zip'):
	if (not os.path.exists(filename.split('.')[0])):
		with zipfile.ZipFile(filename, 'r') as zip_ref:
			zip_ref.extractall(os.getcwd())
		try:
			shutil.rmtree('__MACOSX')
		except OSError:
			pass

def preprocess(text):
	global stopwords, stemmer
	# Decode utf8 to ascii
	# Removing headers; Done manually by identifying what is the last line in headers
	text = text.decode('utf8', 'ignore')[text.index('\n\n'):]
	# print text[:10]
	# text = text[text.index('\n') + 1:]
	try:
		# Tokenize into words
		text = nltk.tokenize.word_tokenize(text)
	except LookupError:
		nltk.download('punkt')
		text = nltk.tokenize.word_tokenize(text)
	# Remove punctuations and newlines
	text = map(lambda x: x.strip(), text)
	text = filter(lambda x: x not in string.punctuation, text)
	# Remove empty entries
	text = filter(bool, text)
	# Convert every word to lower case
	text = map(lambda x: x.lower(), text)
	# Remove stray punctuations from text
	for punc in string.punctuation:
		if (punc != '\'' or punc != '"'):
			for i in xrange(len(text)):
				text[i] = text[i].replace(punc, ' ')
	# Get final cleaned text
	text = filter(bool, ' '.join(text).split(' '))
	# Remove stop words
	try:
		text = filter(lambda x: x not in stopwords.words('english'), text)
	except LookupError:
		nltk.download('stopwords')
		text = filter(lambda x: x not in stopwords.words('english'), text)
	# Apply stemming
	text = map(lambda x: stemmer.stem(x), text)
	return text

def generate_unigram_inverted_index(dir_path = '20_newsgroups'):
	unigram_inverted_index = {}
	for dir_ in os.listdir(dir_path):
		print 'Topic :', dir_
		try:
			for file_ in os.listdir(os.path.join(dir_path, dir_)):
				with open(os.path.join(dir_path, dir_, file_), 'r') as f:
					text = preprocess(f.read())
					for token in text:
						token_ = token.encode('ascii', 'ignore')
						unigram_inverted_index[token_] = unigram_inverted_index.get(token_, []) + [os.path.join(dir_, file_)]
			with open('unigram_inverted_index.json', 'w') as f:
				str_ = json.dumps(unigram_inverted_index)
				json.dump(str_, f)
		except OSError:
			pass

if __name__ == '__main__':
	args = parser.parse_args()
	if (args.unzip):
 		unzip(args.unzip)
 	if (args.generate_unigram_inverted_index):
 		generate_unigram_inverted_index(args.generate_unigram_inverted_index)
