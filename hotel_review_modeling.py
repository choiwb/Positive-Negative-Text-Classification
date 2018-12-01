import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn import metrics
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from nltk.stem import SnowballStemmer
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
import time
from mlxtend.plotting import plot_confusion_matrix

data = pd.read_csv('hotel_review.csv')
data = data[['text','polarity']]

# 데이터 전처리
def clean_text(text):
    # 불필요한 구두점 제거 제거
    text = text.translate(string.punctuation)
    # 모든 알파벳 소문자화
    text = text.lower().split()
    # 모델링 성능 향상을 위한 불필요한 단어 제거
    stops = set(stopwords.words("english"))
    # 불필요한 단어 추가 제거
    new_stopwords = ['chicago', 'hotel', 'hotels', 'room', 'rooms']
    final_stopwords = stops.union(new_stopwords)
    # 한 단어당 3글자 이상 단어만 훈련
    text = [w for w in text if not w in final_stopwords and len(w) >= 3]
    text = " ".join(text)
    text = text.split()
    stemmer = SnowballStemmer('english')
    # 형태소 분석
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

data['text'] = data['text'].map(lambda x: clean_text(x))

data['polarity'] = data['polarity'].map(lambda x : 1 if x == 'positive' else 0)

# print(len(data['polarity'] == 'positive'))
# print(len(data['polarity'] == 'negative'))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_features = 10000

tokenizer = Tokenizer(nb_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

start = time.time()

###########################################################################################
'''
# Only LSTM
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
'''
###########################################################################################

###########################################################################################
# CNN + LSTM

model = Sequential()
model.add(Embedding(max_features, 100, input_length=X.shape[1]))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize defined model
print(model.summary())

###########################################################################################

Y = pd.get_dummies(data['polarity']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.4, random_state = 0)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 32
model.fit(X_train, Y_train, epochs = 2, batch_size = batch_size)

# validation split = 0.2
validation_size = 320

X_validate = X_test[-validation_size:]
print(X_validate)
Y_validate = Y_test[-validation_size:]
print(Y_validate)
X_test = X_test[:-validation_size]
print(X_test)
Y_test = Y_test[:-validation_size]
print(Y_test)

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

end = time.time()
operation = end - start
print('훈련 시간 : %.4f (초)' %(operation))

pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):

    result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=batch_size, verbose=2)[0]

    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("pos_acc: %.4f" %(pos_correct / pos_cnt * 100), "%")
print("neg_acc: %.4f" %(neg_correct / neg_cnt * 100), "%")

y_pred = model.predict(X_test)

# Confusion Matrix의 4가지 결과값 도출
cm = metrics.confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1))

tp = cm[0,0]
tn = cm[1,1]
fp = cm[0,1]
fn = cm[1,0]

acc = (tp + tn) / (tp + tn + fp + fn)
prec = tp / (tp+fp)
sen = tp / (tp+fn)
spec = tn / (fp + tn)

print('정확도 (Accuracy): %.4f , 정밀도 (Precision): %.4f , 민감도 (Sensitivity): %.4f , 특이도 (Specificity): %.4f' % (acc, prec, sen, spec))

binary = np.array([[tn,fp], [fn,tp]])
fig, ax = plot_confusion_matrix(conf_mat = binary, show_absolute=True, show_normed=True, colorbar=True)
plt.show()

# ROC 커브
y_pred_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = metrics.roc_curve(Y_test.argmax(axis = 1), y_pred_prob)

plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
_ = plt.xlabel('False Positive (1 - Specificity)')
_ = plt.ylabel('True Positive (Sensitivity)')
_ = plt.title('ROC Curve')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
plt.show()

# AUC 도출
# print('AUC 넓이 : ', metrics.roc_auc_score(Y_test.argmax(axis = 1), y_pred_prob))
print('AUC 넓이 : %.4f' %(metrics.roc_auc_score(Y_test.argmax(axis = 1), y_pred_prob)))
