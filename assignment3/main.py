from collections import Counter
import argparse
import itertools
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import sklearn
import string
import zipfile

parser = argparse.ArgumentParser(description='Naive Bayes text classifier')
parser.add_argument('--split_ratio', help='Specify the train to test split ratio', type=float, default=0.2)
parser.add_argument('--plot_CM', help='Plot the CM curve', default=False, action='store_true')

stopwords = nltk.corpus.stopwords
stemmer = nltk.stem.snowball.SnowballStemmer('english')


def unzip(filename='20_newsgroups.zip'):
    if (not os.path.exists(filename.split('.')[0])):
        with zipfile.ZipFile(filename, 'r') as f:
            f.extractall(os.getcwd())


def preprocess(text):
    global stopwords, stemmer
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # Decode utf8 to ascii
    text = text.decode('utf8', 'ignore')
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


def relevant_docs(split_ratio, root_dir='20_newsgroups'):
    relevant_folders = [
        'sci.med',
        'sci.space',
        'rec.sport.hockey',
        'talk.politics.misc',
        'comp.graphics']
    folders = map(lambda x: os.path.join(root_dir, x), relevant_folders)
    class_vocabs = [[] for _ in xrange(len(relevant_folders))]
    vocab = []
    # Iterate over each class and corresponding doclist
    print 'Training Naive Bayes classifier'
    for class_, relevant_folder in enumerate(folders):
        docs = sorted(os.listdir(relevant_folder))
        for i, doc in enumerate(docs):
            # Make classifier on train data
            if (i <= (1 - split_ratio) * len(docs)):
                path = os.path.join(relevant_folder, doc)
                with open(path, 'r') as f:
                    text = preprocess(f.read())
                    vocab += text
                    class_vocabs[class_] += text
    return Counter(vocab), map(lambda x: Counter(x), class_vocabs)


def test(split_ratio, plot_CM=False, root_dir='20_newsgroups'):
    vocab, class_vocabs = relevant_docs(split_ratio)
    relevant_folders = [
        'sci.med',
        'sci.space',
        'rec.sport.hockey',
        'talk.politics.misc',
        'comp.graphics']
    folders = map(lambda x: os.path.join(root_dir, x), relevant_folders)
    print 'Testing Naive Bayes classifier'
    predictions, GT = [], []
    for class_, relevant_folder in enumerate(folders):
        docs = sorted(os.listdir(relevant_folder))
        for i, doc in enumerate(docs):
            if (i > (1 - split_ratio) * len(docs)):
                path = os.path.join(relevant_folder, doc)
                with open(path, 'r') as f:
                    text = preprocess(f.read())
                    mx = -np.inf
                    prediction = -1
                    for j in xrange(len(relevant_folders)):
                        log_probs = map(lambda x: np.log(class_vocabs[j].get(x, 0) + 1), text)
                        log_probs = map(lambda x: x - np.log(len(vocab) + len(class_vocabs[j])), log_probs)
                        if (mx < np.sum(log_probs)):
                            prediction = j
                            mx = np.sum(log_probs)
                predictions.append(prediction)
                GT.append(class_)
    # import pdb;pdb.set_trace()
    if (plot_CM):
        conf_matrix = sklearn.metrics.confusion_matrix(GT, predictions)
        cmap = plt.cm.Blues
        plt.imshow(conf_matrix, cmap=cmap)
        plt.title('Confusion matrix with train-test ratio {}-{}'.format(100 - 100 * split_ratio, 100 * split_ratio))
        plt.colorbar()
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        thresh = conf_matrix.max() / 2.
        for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")
        plt.show()
    return 100. * sklearn.metrics.accuracy_score(GT, predictions)


if __name__ == '__main__':
    args = parser.parse_args()
    unzip()
    print 'Accuracy: {}%'.format(test(args.split_ratio, args.plot_CM))
