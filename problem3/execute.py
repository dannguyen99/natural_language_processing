import re
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import metrics
from collections import defaultdict
import math


def load_data(file_path):
    data = []
    # Regular expression to get the label and the text
    regx = re.compile(r'^(\+1|-1)\s+(.+)$')
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            match = regx.match(line)
            if match:
                lb = match.group(1)
                text = match.group(2)
                data.append((text, lb))
    return data


data = load_data('./sentiment.txt')

texts, labels = zip(*data)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

import string


def build_vocab(texts):
    """Build vocabulary from dataset

    Args:
        texts (list): list of tokenized sentences

    Returns:
        vocab (dict): map from word to index
    """
    vocab = {}
    for s in texts:
        for word in s.split():
            # Check if word is a punctuation
            if word in string.punctuation:
                continue
            if word not in vocab:
                idx = len(vocab)
                vocab[word] = idx
    return vocab


def train_naive_bayes(texts, labels, target_classes, alpha=1):
    """Train a multinomial Naive Bayes model
    """
    ndoc = 0
    nc = defaultdict(int)  # map from a class label to number of documents in the class
    logprior = dict()
    loglikelihood = dict()
    count = defaultdict(int)  # count the occurrences of w in documents of class c

    # texts = build_not_text(texts)
    vocab = build_vocab(texts)
    # Training
    for s, c in zip(texts, labels):
        ndoc += 1
        nc[c] += 1
        for w in s.split():
            if w in vocab:
                count[(w, c)] += 1

    vocab_size = len(vocab)
    for c in target_classes:
        logprior[c] = math.log(nc[c] / ndoc)
        sum_ = 0
        for w in vocab.keys():
            if (w, c) not in count:
                count[(w, c)] = 0
            sum_ += count[(w, c)]

        for w in vocab.keys():
            loglikelihood[(w, c)] = math.log((count[(w, c)] + alpha) / (sum_ + alpha * vocab_size))

    return logprior, loglikelihood, vocab


# data = [
#     ("Chinese Beijing Chinese", "c"),
#     ("Chinese Chinese Shanghai", "c"),
#     ("Chinese Macao", "c"),
#     ("Tokyo Japan Chinese", "j")
# ]
# texts, labels = zip(*data)
# target_classes = ["c", "j"]
#
# logprior, loglikelihood, vocab = train_naive_bayes(texts, labels, target_classes)
#
# assert logprior['c'] == math.log(0.75)
# assert logprior['j'] == math.log(0.25)
# assert loglikelihood[('Chinese', 'c')] == math.log(3 / 7)
# assert loglikelihood[('Tokyo', 'c')] == math.log(1 / 14)
# assert loglikelihood[('Japan', 'c')] == math.log(1 / 14)
# assert loglikelihood[('Tokyo', 'j')] == math.log(2 / 9)


def test_naive_bayes(testdoc, logprior, loglikelihood, target_classes, vocab):
    sum_ = {}
    print(testdoc)
    for c in target_classes:
        already = []
        sum_[c] = logprior[c]
        for w in testdoc.split():
            if w in vocab and w not in already:
                print(w)
                sum_[c] += loglikelihood[(w,c)]
                already.append(w)
    # sort keys in sum_ by value
    sorted_keys = sorted(sum_.keys(), key=lambda x: sum_[x], reverse=True)
    return sorted_keys[0]


def build_not_text(texts):
    logical_negation = ["n't", "not", "no", "never"]
    new_text = []
    for s in texts:
        doc = s.split()
        for word_index in range(len(doc)):
            if doc[word_index] in logical_negation:
                for after_not_index in range(word_index + 1, len(doc)):
                    if doc[after_not_index] in string.punctuation:
                        break
                    doc[after_not_index] = "NOT_" + doc[after_not_index]
        new_text.append(' '.join(doc))
    return new_text


def build_vocab_with_not(texts):
    """Build vocabulary from dataset

    Args:
        texts (list): list of tokenized sentences

    Returns:
        vocab (dict): map from word to index
    """
    vocab = {}
    logical_negation = ["n't", "not", "no", "never"]
    for s in texts:
        doc = s.split()
        for word_index in range(len(doc)):
            if doc[word_index] in logical_negation:
                for after_not_index in range(word_index, len(doc)):
                    if doc[after_not_index] in string.punctuation:
                        break
                    doc[after_not_index] = "NOT_" + doc[after_not_index]
            # Check if word is a punctuation
            if doc[word_index] in string.punctuation:
                continue
            if doc[word_index] not in vocab:
                idx = len(vocab)
                vocab[doc[word_index]] = idx
    return vocab


def bool_train_naive_bayes(texts, labels, target_classes, alpha=1):
    """Train a boolean multinomial Naive Bayes model
    """
    ndoc = 0
    nc = defaultdict(int)  # map from a class label to number of documents in the class
    logprior = dict()
    loglikelihood = dict()
    count = defaultdict(int)  # count the occurrences of w in documents of class c

    texts = build_not_text(texts)
    vocab = build_vocab(texts)

    # Training
    for s, c in zip(texts, labels):
        ndoc += 1
        nc[c] += 1
        already = []
        for w in s.split():
            if w in vocab and w not in already:
                count[(w, c)] += 1
                already.append(w)

    vocab_size = len(vocab)
    for c in target_classes:
        logprior[c] = math.log(nc[c] / ndoc)
        sum_ = 0
        for w in vocab.keys():
            if (w, c) not in count: count[(w, c)] = 0
            sum_ += count[(w, c)]

        for w in vocab.keys():
            loglikelihood[(w, c)] = math.log((count[(w, c)] + alpha) / (sum_ + alpha * vocab_size))

    return logprior, loglikelihood, vocab


target_classes = ['+1', '-1']  # we can construct a fixed set of classes from train_labels
logprior, loglikelihood, vocab = bool_train_naive_bayes(train_texts, train_labels, target_classes)

predicted_labels = [test_naive_bayes(s, logprior, loglikelihood, target_classes, vocab)
                    for s in test_texts]

print('Accuracy score: %f' % metrics.accuracy_score(test_labels, predicted_labels))

print(metrics.classification_report(test_labels, predicted_labels))
