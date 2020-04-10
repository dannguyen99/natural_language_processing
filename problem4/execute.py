def load_data(input_file):
    data = []
    with open(input_file, "r") as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line == '':
                data.append(sentence)
                sentence = []
            elif line == 'B-W' or line == "I-W":
                sentence.append((" ", line))
            else:
                word, tag = line.split('\t')
                sentence.append((word, tag))
    return data


train_data = load_data('./train.txt')
# test_data = load_data('./test.txt')

print(train_data[0])


def feature_func(word, previous_tag):
    # w1, w2, .., w_m

    feat_vec = {}

    # Indicator (boolean) features
    # Nguyễn Văn A
    if word[0].isupper():
        feat_vec['is_first_capital'] = 1
    # IBM
    if int(word.upper() == word):
        feat_vec['is_all_caps'] = 1
    if word.lower() == word:
        feat_vec['is_all_lower'] = 1
    if '-' in word:
        feat_vec['has_hyphen'] = 1
    # 1, 2, 345
    if word.isdigit():
        feat_vec['is_numeric'] = 1
    # aBC
    if word[1:].lower() != word[1:]:
        feat_vec['capitals_inside'] = 1

    # Lexical features
    feat_vec['word'] = word
    feat_vec['word.lower()'] = word.lower()
    feat_vec['prefix_1'] = word[0]
    feat_vec['prefix_2'] = word[:2]
    feat_vec['prefix_3'] = word[:3]
    feat_vec['prefix_4'] = word[:4]
    feat_vec['suffix_1'] = word[-1]
    feat_vec['suffix_2'] = word[-2:]
    feat_vec['suffix_3'] = word[-3:]
    feat_vec['suffix_4'] = word[-4:]

    # Tag feature
    feat_vec['previous_tag'] = previous_tag

    return feat_vec


training_instances = []
training_labels = []
for sen in train_data:
    for i, (word, tag) in enumerate(sen):
        if i == 0:
            previous_tag = '<bos>'  # special tag for the beginning of a sentence
        else:
            previous_tag = sen[i - 1][1]
        training_instances.append((word, previous_tag))
        training_labels.append(tag)

list(zip(training_instances[:10], training_labels[:10]))

# prefix_1 + 'đ'
# prefix_2 + 'đư'
# 'word=đường'
# print(feature_func('đường', 'P'))
#
train_feat_vecs = [feature_func(word, tag) for word, tag in training_instances]
from sklearn.feature_extraction import DictVectorizer

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(train_feat_vecs)
X_train.shape
#
from sklearn.linear_model import LogisticRegression

maxent = LogisticRegression(max_iter=1000)
maxent.fit(X_train, training_labels)
#
# print(maxent.predict(vectorizer.transform([feature_func('ngắn', 'B-W')])))
# print(maxent.classes_)


# Since the output only considers indexes, we need to use a map from label to index
target_classes = maxent.classes_.tolist()
label2id = {k: v for v, k in enumerate(target_classes)}
print(label2id)


def get_negative_log_proba(word, previous_tag):
    prob = maxent.predict_log_proba(vectorizer.transform([feature_func(word, previous_tag)]))
    return -prob[0]


#
#
def decode(words):
    """Get the tag sequence for a word sequence
    """
    l = len(words)
    best_score = {}
    best_edge = {}
    best_score[('0 <bos>')] = 0  # Start with <s>
    best_edge[('0 <bos>')] = None

    # Forward Step

    # For the beginning position, the previous tag is <bos>
    w1 = words[0]
    neg_log_proba1 = get_negative_log_proba(w1, '<bos>')
    for tag in target_classes:
        idx = label2id[tag]
        best_score[str(1) + ' ' + tag] = neg_log_proba1[idx]
        best_edge[str(1) + ' ' + tag] = '0 <bos>'

    for i in range(1, l):
        # Calculate:
        # best_score[str(i+1) + ' ' + tag] and
        # best_edge[str(i+1) + ' ' + tag]
        w = words[i]
        for prev_tag in target_classes:
            neg_log_proba = get_negative_log_proba(w, prev_tag)
            for next_tag in target_classes:
                idx = label2id[next_tag]
                if str(i) + ' ' + prev_tag in best_score:
                    score = best_score[str(i) + ' ' + prev_tag] + neg_log_proba[idx]

                    if str(i + 1) + " " + next_tag not in best_score or best_score[str(i + 1) + " " + next_tag] > score:
                        best_score[str(i + 1) + " " + next_tag] = score
                        best_edge[str(i + 1) + " " + next_tag] = str(i) + " " + prev_tag

    # The for <eos>
    for prev in target_classes:
        if str(l) + ' ' + prev in best_score:
            score = best_score[str(l) + ' ' + prev]
            if str(l + 1) + ' ' + '<eos>' not in best_score or best_score[str(l + 1) + ' <eos>'] > score:
                best_score[str(l + 1) + ' <eos>'] = score
                best_edge[str(l + 1) + ' <eos>'] = str(l) + ' ' + prev

    # Backward Step
    tags = []
    next_edge = best_edge[str(l + 1) + " " + "<eos>"]
    while next_edge != "0 <bos>":
        position, tag = next_edge.split()
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags


# N N V CH N N NNP A CH
print(decode('Tôi là sinh viên . Tôi là giảng viên đại học .'.split(' ')))
# print(
#     decode(['Bước', 'vào', 'phòng', 'là', 'một', 'bà_cụ', 'hom_hem', ',', 'đôi', 'mắt', 'vẫn', 'còn', 'tinh_anh', '.']))
#
#
# def flat_accuracy(test_data):
#     correct = 0
#     total = 0
#     for i, sen in enumerate(test_data):
#         words, gold_tags = zip(*sen)
#         predicted_tags = decode(words)
#         # print(i+1, gold_tags, predicted_tags)
#         if (i + 1) % 200 == 0:
#             print('Predicted %d/%d sentences' % (i + 1, len(test_data)))
#         for t1, t2 in zip(predicted_tags, gold_tags):
#             total += 1
#             if t1 == t2:
#                 correct += 1
#     return 100.0 * correct / total, correct, total
