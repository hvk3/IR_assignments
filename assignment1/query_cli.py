import argparse
import json
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import time
from wordcloud import WordCloud

# Handle queries now
# Everything prior done

parser = argparse.ArgumentParser(description = 'Utilities for preprocessing documents')
parser.add_argument('--display_wordcloud', help = 'Display the wordcloud of the inverted index'
	, default = None, action = 'store_true')
parser.add_argument('--store_wordcloud', help = 'Store the wordcloud of the inverted index in the give filename'
	, default = None, action = 'store_true')
parser.add_argument('--load_unigram_inverted_index', type = str, help = 'Load a previously generated inverted index from the\
	given json file', default = 'unigram_inverted_index.json')
parser.add_argument('--process_query', type = str, nargs = '+', help = 'Handle queries of the form X <AND/OR> <NOT> Y, in any case;\
	X and Y can themselves be queries to be processed', default = None)
parser.add_argument('--use_skip_lists', type = int, help = 'Use skip lists with the specified skip distance', default = 1)
parser.add_argument('--test_performance', help = 'Test performance with 1000 random keys', default = None, action = 'store_true')

def get_unigram_inverted_index(json_file = 'unigram_inverted_index.json'):
	with open(json_file, 'r') as f:
		unigram_inverted_index = json.loads(json.load(f))
	try:
		assert(unigram_inverted_index[unigram_inverted_index.keys()[0]]\
			== sorted(unigram_inverted_index[unigram_inverted_index.keys()[0]]))
	except AssertionError:
		for token in unigram_inverted_index:
			unigram_inverted_index[token] = sorted(unigram_inverted_index[token])
		import pdb;pdb.set_trace()
		with open(json_file, 'w') as f:
			str_ = json.dumps(unigram_inverted_index)
			json.dump(str_, f)
	return unigram_inverted_index

def generate_wordcloud(unigram_inverted_index, display_wordcloud = False, save_wordcloud = False):
	# Reference : https://github.com/amueller/word_cloud/blob/master/examples/simple.py
	if (not save_wordcloud and not display_wordcloud):
		return
	wordcloud_dict = {i : len(unigram_inverted_index[i]) for i in unigram_inverted_index}
	wordcloud_ = WordCloud(width = 2000, height = 2000).generate_from_frequencies(wordcloud_dict)
	plt.axis('off')
	if (save_wordcloud):
		plt.imshow(wordcloud_, interpolation = 'bilinear')
		plt.savefig('wordcloud.png')
	if (display_wordcloud):
		plt.imshow(wordcloud_, interpolation = 'bilinear')
		plt.show()

def get_all_docs(dir_path = '20_newsgroups'):
	all_docs = []
	for dir_ in os.listdir(dir_path):
		try:
			for file in os.listdir(os.path.join(dir_path, dir_)):
				all_docs.append(file)
		except OSError:
			pass
	return all_docs

def merge(postingslists, query_type, skip_distance = 1):
	# import pdb;pdb.set_trace()
	skip_counter = 0
	for i in xrange(len(postingslists) - 1):
		combined_postingslist = []
		p1 = postingslists[i]
		p2 = postingslists[i + 1]
		j, k = 0, 0
		if query_type == 'and':
			while (j < len(p1) and k < len(p2)):
				flag = False
				if (p1[j] == p2[k]):
					combined_postingslist.append(p1[j])
					j += 1
					k += 1
				elif (p1[j] < p2[k]):
					if (j % skip_distance == 0):
						skipped_j = j + skip_distance
						if (skipped_j >= len(p1)):
							j += 1
						elif (skipped_j < len(p1) and p1[skipped_j] < p2[k]):
							j = skipped_j
							skip_counter += 1
						elif (p1[skipped_j] > p2[k]):
							j += 1
						elif (p1[skipped_j] == p2[k]):
							combined_postingslist.append(p2[k])
							j = skipped_j + 1
							skip_counter += 1
							k += 1
					else:
						j += 1
				elif (p1[j] > p2[k]):
					if (k % skip_distance == 0):
						skipped_k = k + skip_distance
						if (skipped_k >= len(p2)):
							k += 1
						elif (skipped_k < len(p2) and p2[skipped_k] < p1[j]):
							k = skipped_k
							skip_counter += 1
						elif (p2[skipped_k] > p1[j]):
							k += 1
						elif (p2[skipped_k] == p1[j]):
							combined_postingslist.append(p1[j])
							j += 1
							skip_counter += 1
							k = skipped_k + 1
					else:
						k += 1
		else:
			combined_postingslist = list(set(p1).union(set(p2)))
		postingslists[i + 1] = combined_postingslist
	# print 'Total skips :', skip_counter
	return postingslists[-1]

def get_results(unigram_inverted_index, keys, query_type = '', skip_distance = 1):
	# import pdb;pdb.set_trace()
	postingslists = []
	all_docs = get_all_docs()
	for idx, key in enumerate(keys):
		key_ = map(lambda x: x.lower(), key.split(' '))
		if ('not' == key_[0]):
			postingslists.append(list(set(all_docs) - set(unigram_inverted_index[key_[1]])))
		else:
			postingslists.append(np.copy(unigram_inverted_index[key_[0]]))
	postingslists = sorted(postingslists, key = lambda x: len(x))
	return merge(postingslists, query_type, skip_distance)

def test_performance(unigram_inverted_index):
	query_type = ['and', 'or']
	# np.random.seed(0)
	random_keys = sorted(unigram_inverted_index.keys(), key = lambda x: len(unigram_inverted_index[x]))[-100:]
	np.save('random_keys', random_keys)
	print random_keys
	two_tuples = [np.random.choice(random_keys, 2, replace = False) for i in xrange(25)]
	for idx, two_tuple in enumerate(two_tuples):
		# print 'Keys:', two_tuple
		two_tuple = list(two_tuple)
		for query in query_type:
			print query,
			for i in xrange(2):
				if (i == 1):
					two_tuple[1] = 'not ' + two_tuple[1]
				print 'Keys:', two_tuple
				if (query == 'and'):
					skip_distances = [1, 3, 5, 10, 50]
				else:
					skip_distances = [1]
				for skip_distance in skip_distances:
					start_time = time.time()
					answer = set(get_results(unigram_inverted_index, two_tuple, query, skip_distance))
					print 'Time taken to process the query : ' + str(time.time() - start_time) + ' seconds'
					print 'Number of relevant documents found :', len(answer)
					# if len(answer) == 0:
					# 	answer = ['None']
					# print 'Documents found :',
					# for doc in answer:
					# 	print doc,

if __name__ == '__main__':
	args = parser.parse_args()
	stemmer = nltk.stem.snowball.SnowballStemmer('english')
	unigram_inverted_index = get_unigram_inverted_index(args.load_unigram_inverted_index)
	generate_wordcloud(unigram_inverted_index, args.display_wordcloud is not None, args.store_wordcloud is not None)
	if (args.process_query):
		start_time = time.time()
		query = ' '.join(args.process_query).lower()
		skip_distance = int(args.use_skip_lists)
		print 'Query searched:', query
		if (' and ' in query):
			query_type = 'and'
			keys = query.split(' and ')
		else:
			query_type = 'or'
			keys = query.split(' or ')
		keys = map(lambda x: stemmer.stem(x), keys)
		answer = set(get_results(unigram_inverted_index, keys, query_type, skip_distance))
		print 'Time taken to process the query : ' + str(time.time() - start_time) + ' seconds'
		print 'Number of relevant documents found :', len(answer)
		if len(answer) == 0:
			answer = ['None']
		print 'Documents found :',
		for doc in answer:
			print doc,
	if (args.test_performance):
		test_performance(unigram_inverted_index)
