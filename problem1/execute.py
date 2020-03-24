from collections import defaultdict
import math


def train_bigram(train_file, model_file):
    """Train trigram language model and save to model file
    """
    counts = defaultdict(int)  # count the n-gram
    context_counts = defaultdict(int)  # count the context
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            words = line.split()
            words.append('</s>')
            words.insert(0, '<s>')

            for i in range(1, len(words)):  # Note: starting at 1, after <s>
                # TODO: Write code to count bigrams and their contexts
                # YOUR CODE HERE
                counts[words[i - 1] + " " + words[i]] += 1
                context_counts[words[i - 1]] += 1
                counts[words[i]] += 1
                context_counts[""] += 1

    # Save probabilities to the model file
    with open(model_file, 'w') as fo:
        for ngram, count in counts.items():
            # TODO: Write code to calculate probabilities of n-grams
            # (unigrams and bigrams)
            # Hint: probabilities of n-grams will be calculated by their counts
            # divided by their context's counts.
            # probability = counts[ngram]/context_counts[context]
            # After calculating probabilities, we will save ngram and probability
            # to the file in the format:
            # ngram<tab>probability

            # YOUR CODE HERE
            words = ngram.split(" ")
            words.pop()
            context = " ".join(words)
            probability = counts[ngram] / context_counts[context]
            fo.write('%s\t%f\n' % (ngram, probability))


def load_bigram_model(model_file):
    """Load the model file

    Args:
        model_file (str): Path to the model file

    Returns:
        probs (dict): Dictionary object that map from ngrams to their probabilities
    """
    probs = {}
    with open(model_file, 'r') as f:
        for line in f:
            # TODO: From each line split ngram, probability
            # and then update probs

            # YOUR CODE HERE

            line = line.strip()
            if line == '':
                continue
            words = line.split()
            if len(words) == 2:
                w, p = line.split()
                w = w.lower()
                probs[w] = float(p)
            elif len(words) == 3:
                w1, w2, p = line.split()
                w1 = w1.lower()
                w2 = w2.lower()
                probs[w1 + " " + w2] = float(p)
    return probs


def test_bigram(test_file, model_file, lambda2=0.95, lambda1=0.95, N=1000000):
    W = 0  # Total word count
    H = 0
    probs = load_bigram_model(model_file)
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            words = line.split()
            words.append('</s>')
            words.insert(0, '<s>')
            for i in range(1, len(words)):  # Note: starting at 1, after <s>
                # TODO: Write code to calculate smoothed unigram probabilties
                # and smoothed bigram probabilities
                # You should use calculate p1 as smoothed unigram probability
                # and p2 as smoothed bigram probability
                p1 = None
                p2 = None

                # YOUR CODE HERE
                p1 = (1 - lambda1) / N
                if words[i] in probs:
                    p1 += lambda1 * probs[words[i].lower()]
                p2 = (1 - lambda2) * p1
                if words[i - 1].lower() + " " + words[i].lower() in probs:
                    p2 += lambda2 * probs[words[i - 1].lower() + " " + words[i].lower()]
                # END OF YOUR CODE
                W += 1  # Count the words
                H += -math.log2(p2)  # We use logarithm to avoid underflow
    H = H / W
    P = 2 ** H
    print("Entropy: {}".format(H))
    print("Perplexity: {}".format(P))
    return P


# train_bigram('02-train-input.txt', '02-train-answer.txt')
# probs = load_bigram_model('bigram_model.txt')
# print(probs)
train_bigram('wiki-en-train.word', 'bigram_model.txt')
test_bigram('wiki-en-test.word', 'bigram_model.txt')
