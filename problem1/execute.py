from collections import defaultdict
import math


def train_bigram(train_file, model_file):
    """Train trigram language model and save to model file
    """
    counts = defaultdict(int)  # count the n-gram
    context_counts = defaultdict(int)  # count the context
    total_count = 0  # to count total words
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
                counts[words[i - 1]] += 1
                context_counts[words[i - 1], words[i]] += 1
                total_count += 1
                if i == len(words) - 1:
                    counts[words[i]] += 1

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
            probability = count / total_count
            fo.write('%s\t%f\n' % (ngram, probability))
            for context, context_count in context_counts.items():
                if ngram == context[0]:
                    probability = context_count / count
                    str_context = ' '.join(context)
                    fo.write('%s\t%f\n' % (str_context, probability))


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
                probs[w1, w2] = float(p)
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
                if (words[i - 1], words[i]) in probs:
                    p1 = lambda1 * probs[words[i].lower()] + (1 - lambda1) / N
                    p2 = lambda2 * probs[words[i - 1].lower(), words[i].lower()] + (1 - lambda2) * p1
                else:
                    continue
                # END OF YOUR CODE
                W += 1  # Count the words
                H += -math.log2(p2)  # We use logarithm to avoid underflow
        H = H / W
        P = 2 ** H
        print("Entropy: {}".format(H))
        print("Perplexity: {}".format(P))
        return P
