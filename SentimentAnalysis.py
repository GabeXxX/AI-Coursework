#%%
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud, STOPWORDS

from scipy import stats

from tools import wm2df
from tools import plotLearningCurve
from tools import evalPredictions
from tools import cloud
from tools import cleaning

#%%
#Load dataset in a panda DataFrame
df = pd.read_json('./Dataset/Toys_and_Games_5.json',lines=True)

#%%
#Look at the data
print("Dimensione dataset: ", df.shape, "\n")
print("Visualize first two rows of the dataset:", "\n")
df.iloc[0:2]
print("Statistiche di ciascun attributo: ", "\n", df.describe(), "\n")
print("Distribuzione dei valori dell'attributo overall: ", "\n", df.groupby("overall").size(), "\n")

#%%
#Modified the structure of dataset for include only columns of our interest
dataset = pd.DataFrame()
dataset["overall"] = df["overall"]
dataset["reviewText"] = df["reviewText"]
dataset["summary"] = df["summary"]
dataset.iloc[0:5]

#%%
#Data cleaning and text preprocessing
#Let's visualize the effect of varius data cleaning technique for a single review

#Remove non-letter characters and convert reviews to lower case
review = dataset.iloc[0]["reviewText"]
print(review)

letter = re.sub("[^a-zA-Z]", " ", review)
letter = letter.lower()
print(letter)

#Tokenization 
word = nltk.word_tokenize(letter)
word

#Remove stop-words
print("List of english stop-words: \n")
print(stopwords.words("english"))
word = [w for w in word if not w in set(stopwords.words("english"))]
print(word)

#Lemmization
snow = nltk.stem.SnowballStemmer("english")
stems = [snow.stem(w) for w in word]
print(stems)

#%%

#Apply data cleaning and text preprocessing to all dataset
#Add the processed data to the original data. 
#Using apply() function is more elegant and concise than using for loop
dataset["cleanReview"] = dataset["reviewText"].apply(cleaning)
dataset["cleanSummary"] = dataset["summary"].apply(cleaning)
dataset.head()

#Save processed data set in order to retrieve it
dataset.to_pickle("./Dataset/processedDataSet.pkl")

#%%
#Retrieve precedent saved dataset
dataset = pd.read_pickle("./Dataset/processedDataSet.pkl")
dataset.head()

#%%
#Visualization

#WordCloud
cloud(' '.join(dataset["cleanReview"]))
cloud(' '.join(dataset["cleanSummary"]))

#%%
#Distribution
# We need to split each words in cleaned review and then count the number of each rows of data frame.

dataset["reviewLenght"] = dataset["cleanReview"].apply(lambda x: len(str(x).split())) #Split() a string into a list where each word is a list item
dataset["summaryLenght"] = dataset["cleanSummary"].apply(lambda x: len(str(x).split()))

sns.kdeplot(dataset["reviewLenght"], shade=True)

print("median of review lenght: ", dataset["reviewLenght"].median())


#%%
dataset['overall'].value_counts().plot(kind='bar', color='cornflowerblue')
dataset.iloc[:2]

#%%
#Create validation and training set
xTrain, xTest, yTrain, yTest = train_test_split(dataset["cleanReview"],
                                                 dataset["overall"],
                                                  test_size=0.2, 
                                                   random_state=1)

#%%
#Build vectorizer
vect = []
vect.append(CountVectorizer(binary = True))
vect.append(CountVectorizer())
vect.append(CountVectorizer(binary = True))
vect.append(CountVectorizer())

#%%
#Create bag of words representation for bernoulli model
xTrainBern = vect[0].fit_transform(xTrain) #Learn the vocabulary dictionary and return term-document matrix.
xTestBern = vect[0].transform(xTest) #Transform documents to document-term matrix. We only call transform not fit_transform due to the risk of overfitting.

#Create bag of words representation for binomial model
xTrainBin = vect[1].fit_transform(xTrain)
xTestBin = vect[1].transform(xTest)

#Convert in bag of words representation the entire dataset
XBern = vect[2].fit_transform(dataset["cleanReview"])
XBin = vect[3].fit_transform(dataset["cleanReview"])
y = dataset["overall"]

#%%
#Shed some light on the bag of words representation returned from the CountVectorizer class
  
#set of documents
documents = ['The quick brown fox.','The the Jumps over the lazy dog!']
#instantiate the vectorizer object
cvec = CountVectorizer(lowercase=False)
#convert the documents into a document-term matrix
wordMatrix = cvec.fit_transform(documents)
#retrieve the terms found in the corpora
featureNames = cvec.get_feature_names()
#create a dataframe from the matrix
wm2df(wordMatrix, featureNames)

#%%
#Build models
models = []
models.append(BernoulliNB())
models.append(MultinomialNB())


#%%
#Train models
models[0].fit(xTrainBern, yTrain)
models[1].fit(xTrainBin, yTrain)


#%%
#Make class predictions
yPredBern = models[0].predict(xTestBern)
yPredBin = models[1].predict(xTestBin)

#%%
#Calculate accuracy, precision, recall, and F-measure of class predictions
print("Prestazioni di Bernoulli: ", "\n")
print("Numero di parole nel train set:", len(vect[0].get_feature_names()), "\n")
evalPredictions(yTest, yPredBern)
print( "\nPrestazioni di Binomiale: ", "\n")
print("Numero di parole nel train set:", len(vect[1].get_feature_names()), "\n")
evalPredictions(yTest, yPredBin)

#%%Learning Curve

plotLearningCurve(estimator = models[0], 
                   title="Learning Curves (Bernoulli Naive Bayes)", 
                    X=XBern, 
                     y = y, 
                      ylim=(0.0, 1.0), 
                       cv=5,
                        n_jobs= -1, 
                         train_sizes=[1, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
plotLearningCurve(estimator = models[1],
                   title= "Learning Curves (Binomial Naive Bayes)", 
                    X = XBin,
                     y = y, 
                      ylim=(0.0, 1.0), 
                       cv=5,
                        n_jobs= -1, 
                         train_sizes=[1, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])


#%%

#%%
#Distribution
# We need to split each words in cleaned review and then count the number of each rows of data frame.
dataset["reviewLenght"] = dataset["cleanReview"].apply(lambda x: len(str(x).split())) #Split() a string into a list where each word is a list item
dataset["summaryLenght"] = dataset["cleanSummary"].apply(lambda x: len(str(x).split()))

sns.distplot(dataset["reviewLenght"], kde=False)

print("Median of review lenght: ", dataset["reviewLenght"].median())

#%%
