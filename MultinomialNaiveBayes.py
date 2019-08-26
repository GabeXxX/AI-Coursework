############################
# MULTINOMIAL NAIVE BAYESS #
############################
#%%
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

%matplotlib inline

#%%
#Read entire dataset in a panda DataFrame
df = pd.read_json('./Dataset/Toys_and_Games_5.json',lines=True)
print(df)
print(df.groupby('overall').describe())

#Let’s take a look at the distribution of scores across reviews
df['overall'].value_counts().plot(kind='bar', color='cornflowerblue')

#%%
#Drop observations containg NaN in review or star rating.
df = df[df['reviewText'].notnull()]
df = df[df['overall'].notnull()]


#%%
#Split into test and training data.
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'],
                                                   df['overall'],
                                                   test_size=0.2, random_state=1)

#%%
#Represent each review as a bag of words for MULTINOMIAL NAIVE BAYES MODEL:
#a count of how many times each word appears in a document.
#Therefore, convert the collection of training reviews
#into a collection of token counts (a document term matrix).

#Tokenize train and test data
#CountVectorizer implements both tokenization and occurrence counting in a single class:
vect = CountVectorizer()

X_train_dtm = vect.fit_transform(X_train)
print("number words in training corpus:", len(vect.get_feature_names()))
X_test_dtm = vect.transform(X_test)

#%%
#Instantiate and train a multinomial naive Bayes model.
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

#%%
#Make class predictions
y_pred = nb.predict(X_test_dtm)

#%%
#Calculate accuracy, precision, recall, and F-measure of class predictions
def eval_predictions(y_test, y_pred):
    print('accuracy:', metrics.accuracy_score(y_test, y_pred))
    print('precision:', metrics.precision_score(y_test, y_pred, average='weighted'))
    print('recall:', metrics.recall_score(y_test, y_pred, average='weighted'))
    print('F-measure:', metrics.f1_score(y_test, y_pred, average='weighted'))
eval_predictions(y_test, y_pred)

#%%
#Take a look at examples where the model is getting it wrong.
# print message text for the first 3 false positives
print('False positives:')
print
for x in X_test[y_test < y_pred][:2]:
    print(x)
    print()

# print message text for the first 3 false negatives
print('False negatives:')
print
for x in X_test[y_test > y_pred][:2]:
    print(x[:500])
    print()
#%%
