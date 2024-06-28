import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X1_train = df['title'].str.lower() # date adev
X2_train = df['content'].str.lower()# date adev

X1_train = X1_train.fillna('').str.lower()
X2_train = X2_train.fillna('').str.lower()

y = df['class']  # date adev

X1_test = test_df['title'].str.lower()
X2_test = test_df['content'].str.lower()

X1_test = X1_test.fillna('').str.lower()
X2_test = X2_test.fillna('').str.lower()


lista_real = ["biziday"]


pipeline1 = Pipeline([('tfidf', TfidfVectorizer( lowercase=True)), ('binarizer', Binarizer()) , ('clf', BernoulliNB(alpha = 1.2))])
pipeline2 = Pipeline([('tfidf', TfidfVectorizer( lowercase=True)), ('binarizer', Binarizer()), ('clf', BernoulliNB(alpha = 1.2))])

pipeline1.fit(X1_train, y)
pipeline2.fit(X2_train, y)

pred_titlu = pipeline1.predict(X1_test)
pred_content = pipeline2.predict(X2_test)

p_titlu = np.array(pred_titlu).astype(bool)
p_content = np.array(pred_content).astype(bool)


output_data = []

for i in range(len(p_titlu)):
    titlu_real = any(word in X1_test[i] for word in lista_real)
    content_real = any(word in X2_test[i] for word in lista_real)

    if( titlu_real or content_real):
        row = {'id': i, 'class': 0}
    elif( p_titlu[i] or p_content[i]):#mai crestem cate putin
        row = {'id': i, 'class': 1}
    else: row = {'id': i, 'class': 0}

    output_data.append(row)

output_df = pd.DataFrame(output_data)

output_df.to_csv('submission1.csv', index = False)