import argparse
from bs4 import BeautifulSoup
from collections import Counter
import json
import nltk
from num2words import num2words
import os
import string
import zipfile

stopwords = nltk.corpus.stopwords
stemmer = nltk.stem.snowball.SnowballStemmer('english')

parser = argparse.ArgumentParser(description='File for preprocessing documents')
parser.add_argument('--doc_indices', type=str, help='Generate document indices from the given document list', default='stories')
parser.add_argument('--title_indices', type=str, help='Generate document indices from the given document list', default='stories')

idf_index, tf_index = {}, {}

def extract(filename = 'stories.zip'):
	if (not os.path.exists(filename.split('.')[0])):
		with zipfile.ZipFile(filename, 'r') as zip_ref:
			zip_ref.extractall(os.getcwd())

# Identifying headers : check for 2 new lines together, or the first occurence when caps end
def preprocess(text):
	global stopwords, stemmer
	text = text.decode('utf8', 'ignore')
	try:
		nltk.tokenize.word_tokenize('')
	except LookupError:
		nltk.download('punkt')
	text = nltk.tokenize.word_tokenize(text)
	text = map(lambda x: x.strip(), text)
	for idx_, word in enumerate(text):
		try:
			word_ = float(word.replace(',', ''))
			word_ = num2words(word_)
			text.pop(idx_)
			text.extend(word_.split(' '))
		except ValueError:
			continue
	text = filter(lambda x: x not in string.punctuation, text)
	text = filter(bool, text)
	text = map(lambda x: x.lower(), text)
	for punc in string.punctuation:
		if (punc != '\'' or punc != '"'):
			for i in xrange(len(text)):
				text[i] = text[i].replace(punc, ' ')
	text = filter(bool, ' '.join(text).split(' '))
	try:
		_ = stopwords.words('english')
	except LookupError:
		nltk.download('stopwords')
	text = filter(lambda x: x not in stopwords.words('english'), text)
	text = map(lambda x: stemmer.stem(x), text)
	return text

def access_titles_from_html(dir_path = 'stories'):
	dict_ = {}
	with open(os.path.join(dir_path, 'index.html'), 'r') as f:
		lines = filter(lambda x: '<TR VALIGN=TOP>' in x, f.readlines())
		for line in lines:
			line = line.strip()
			if ('</TD></TD></TD></TR>' not in line):
				line += '</TD></TD></TD></TR>'
			soup = BeautifulSoup(line, 'lxml')
			if (soup.find('a').text == 'FARNON' or soup.find('a').text == 'SRE'):
				continue
			dict_[soup.find('a').text] = soup.find_all('td')[-1].text
	return dict_

def dump_json(filename, index):
	with open(filename, 'w') as f:
		json.dump(json.dumps(index), f)

def compute_tf_idf_indices(text, dict_entry):
	global tf_index, idf_index
	text = map(lambda x: x.encode('ascii', 'ignore'), preprocess(text))
	word_counts = Counter(text)
	for token in text:
		idf_index[token] = idf_index.get(token, []) + [dict_entry]
	for token in word_counts:
		if (token not in tf_index):
			tf_index[token] = {}
		tf_index[token][dict_entry] = word_counts[token]

def doc_indices(dir_path = 'stories'):
	global tf_index, idf_index
	for idx, entry in enumerate(os.listdir(dir_path)):
		print idx, entry
		if (os.path.isdir(os.path.join(dir_path, entry)) or 'index' in entry):
			continue
		try:
			with open(os.path.join(dir_path, entry), 'r') as f:
				text = f.read()
				compute_tf_idf_indices(text, entry)
		except OSError:
			pass
	for key in idf_index:
		idf_index[key] = list(set(idf_index[key]))
	dump_json('doc_idf_index.json', idf_index)
	dump_json('doc_tf_index.json', tf_index)

def title_indices(dir_path = 'stories'):
	dict_ = access_titles_from_html(dir_path)
	global tf_index, idf_index
	for doc in dict_:
		text = dict_[doc]
		compute_tf_idf_indices(text, doc)
	for key in idf_index:
		idf_index[key] = list(set(idf_index[key]))
	dump_json('titles_idf_index.json', idf_index)
	dump_json('titles_tf_index.json', tf_index)

if __name__ == '__main__':
	extract()
	args = parser.parse_args()
	if (args.doc_indices):
 		doc_indices(args.doc_indices)
 	if (args.title_indices):
 		title_indices(args.title_indices)
