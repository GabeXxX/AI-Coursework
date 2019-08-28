#########################################
# BERNOULLI AND MULTINOMIAL NAIVE BAYES #
#########################################
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve

from tools import wm2df
from tools import plotLearningCurve
from tools import evalPredictions

#%%
#Load dataset in a panda DataFrame
dataset = pd.read_json('./Dataset/Toys_and_Games_5.json',lines=True)

#%%
#Look at the data
print("Dimensione dataset: ", dataset.shape, "\n")
print("Prime righe del dataset: ", "\n", dataset.head(10), "\n")
print("Statistiche di ciascun attributo: ", "\n", dataset.describe(), "\n")
print("Distribuzione dei valori dell'attributo overall: ", "\n", dataset.groupby("overall").size(), "\n")

#%%
#Visualize the data
dataset['overall'].value_counts().plot(kind='bar', color='cornflowerblue')

#%%
#Create validation and training set
xTrain, xTest, yTrain, yTest = train_test_split(dataset['reviewText'],
                                                   dataset['overall'],
                                                   test_size=0.2, random_state=1)

#%%
#Build models
models = []
models.append(BernoulliNB())
models.append(MultinomialNB())



#Extract feature from dataset:
#represent each review as a feature vector for NAIVE BAYES MODEL, using the BAG OF WORDS representation.
#Vectorize train and test data
#CountVect implements both tokenization and occurrence counting in a single class.
#CountVector is a matrix notation of the dataset in which every row represents a document 
#from the corpus, every column represents a term from the corpus, and every cell 
#represents the frequency count of a particular term in a particular document.
#It also possible to perform binary count(1 if the word occour in a particular document,
#0 otherwise) for Bernoully bayes model.
#Numerous parameters permit to built different bag of words representation
#In this way is possible to visualize how models behaviur change with different representation

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
xTestBern = vect[0].transform(xTest) #Transform documents to document-term matrix.

#Create bag of words representation for binomial model
xTrainBin = vect[1].fit_transform(xTrain)
xTestBin = vect[1].transform(xTest)

#Convert in bag of words representation the entire dataset
XBern = vect[2].fit_transform(dataset["reviewText"])
XBin = vect[3].fit_transform(dataset["reviewText"])
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

#%%
#Take a look at examples where the Bernoulli model is getting it wrong.

#print message text for the first 2 false positives
print("False positives: \n")
for x in xTest[yTest < yPredBern][:2]:
    print(x, "\n")

#print message text for the first 2 false negatives
print("False negatives: \n")
for x in xTest[yTest > yPredBern][:2]:
    print (x, "\n")

#%%
#Generate learning curve

#generate learning curve for bernoulli model
trainSizes, trainScores, validationScores = learning_curve(estimator = models[0], 
                                                            X = XBern,
                                                             y = y, 
                                                              train_sizes=[1, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000], 
                                                               cv = 5,
                                                                scoring = 'accuracy',
                                                                 n_jobs=-1)

#Print scores
print("Numero di parole nel train set:", len(vect[2].get_feature_names()), "\n")
print('Training scores:\n\n', trainScores)
print('\nValidation scores:\n\n', validationScores)

#To plot the learning curves, we need only a single error score 
#per training set size, not 5. For this reason, in the next code
#cell we take the mean value of each row and also flip the signs of the
#error scores (as discussed above).

trainScoresMean = trainScores.mean(axis = 1)
validationScoresMean = validationScores.mean(axis = 1)
print('Mean training scores\n\n', pd.Series(trainScoresMean, index = trainSizes))
print('\nMean validation scores\n\n',pd.Series(validationScoresMean, index = trainSizes))

#Plot
plt.style.use('seaborn')
plt.plot(trainSizes, trainScoresMean, label = 'Training error')
plt.plot(trainSizes, validationScoresMean, label = 'Validation error')
plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for Bernoulli Naive Bayes model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,1)

#%%
#Use a more compact function for plotting learning curve of both model

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
