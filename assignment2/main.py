import argparse
from autocorrect import spell
import json
import numpy as np
import os
from scipy.spatial.distance import cosine

from preprocess import preprocess, access_titles_from_html

parser = argparse.ArgumentParser(description='Main file for IR assignment 2.')
parser.add_argument('--use_raw_tf_idf', help='Use only tf-idf scores for finding relevant documents', default=False, action='store_true')
parser.add_argument('--use_vector_space_tf_idf', help='Use tf-idf based vector space scores for finding relevant documents', default=False, action='store_true')
parser.add_argument('--cache_queries', help='Enable query caching with the specified cache size', default=False, action='store_true')
parser.add_argument('query', type=str, help='The query for which matching documents are to be retrieved')
parser.add_argument('--top_k', type=int, help='Number of documents to return')

def tf_score(token, doc, tf_index):
	docs_and_freqs = tf_index.get(token, {})
	if (doc not in docs_and_freqs):
		return 0
	return 1 + np.log10(docs_and_freqs[doc])

def idf_score(token, idf_index, N):
	return np.log10(N)  - np.log10(len(idf_index.get(token, ['_'])))

def tf_idf_weights(token, doc, title, indices, N, title_weightage):
	idf_score_doc = idf_score(token, indices[0], N)
	tf_score_doc = tf_score(token, doc, indices[1])
	idf_score_title = idf_score(token, indices[2], N)
	tf_score_title = tf_score(token, title, indices[3])
	return (1 - title_weightage) * tf_score_doc * idf_score_doc + title_weightage * tf_score_title * idf_score_title

def get_doc_vector(doc, indices, N, title_weightage = 0.6):
	dict_ = access_titles_from_html()
	title = dict_[doc]
	vec_ = {}
	for token in indices[0]:
		vec_[token] = tf_idf_weights(token, doc, title, indices, N, title_weightage)
	return vec_

def use_raw_tf_idf(query, indices, total_docs, title_weightage=0.6, k=5):
	query_ = preprocess(' '.join(map(lambda x: spell(x), query.split(' '))))
	N = len(total_docs)
	dict_ = access_titles_from_html()
	scores = {}
	for doc in total_docs:
		title = dict_[doc]
		score = 0
		for token in query_:
			score += tf_idf_weights(token, doc, title, indices, N, title_weightage)
		scores[doc] = score
	top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
	return top_k

def use_vector_space_tf_idf(query, indices, total_docs, title_weightage=0.6, k=5):
	query_ = preprocess(' '.join(map(lambda x: spell(x), query.split(' '))))
	print query_
	N = len(total_docs)
	try:
		doc_vectors = np.load('doc_vectors.npy')
	except IOError:
		doc_vectors = map(lambda x: get_doc_vector(x, indices, N, title_weightage), total_docs)
		np.save('doc_vectors', doc_vectors)
	cosine_similarities, query_vec = {}, {}
	for idx, doc in enumerate(total_docs):
		title = dict_[doc]
		query_vec = {}
		for token in indices[0]:
			if (token not in query_):
				query_vec[token] = 0.
			else:
				query_vec[token] = tf_idf_weights(token, doc, title, indices, N, title_weightage)
		u = map(lambda x: x[1], sorted(query_vec.items(), key = lambda x: x[0]))
		v = map(lambda x: x[1], sorted(doc_vectors[idx].items(), key = lambda x: x[0]))
		if (np.max(u) == 0. or np.max(v) == 0.):
			cosine_similarities[doc] = 0
		else:
			cosine_similarities[doc] = 1 - cosine(u, v)
	top_k = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)[:k]
	return top_k

if __name__ == '__main__':
	args = parser.parse_args()
	dict_ = access_titles_from_html()
	max_entries = 20
	try:
		indices = []
		for file in sorted(filter(lambda x: 'json' in x, os.listdir(os.getcwd()))):
			print file
			indices.append(json.loads(json.load(open(file, 'r'))))
	except IOError:
		print 'Generate the indices first!'
		exit()
	total_docs = filter(lambda x: not os.path.isdir(os.path.join('stories', x)) and 'html' not in x, os.listdir('stories'))
	if (args.use_raw_tf_idf):
		method = use_raw_tf_idf
		label = 'raw_'
	elif (args.use_vector_space_tf_idf):
		method = use_vector_space_tf_idf
		label = 'cosine_'
	try:
		cache = np.load(label + 'cache.npy')
	except IOError:
		cache = {}
	if (args.cache_queries):
		if (args.query in cache):
			results = cache[args.query]
		else:
			results = method(args.query, indices, total_docs)
			cache[args.query] = results
			np.save('cache', cache)
	else:
		results = method(args.query, indices, total_docs)
	print results
