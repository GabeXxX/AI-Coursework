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
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from wordcloud import WordCloud, STOPWORDS
from scipy import stats


#%%
#Load dataset in a panda DataFrame
df = pd.read_json('./Dataset/Toys_and_Games_5.json',lines=True)

#%%
#Look at the data
print("Dataset dimension: ", df.shape, "\n")
df.iloc[0:3]

#%%
# Better organize the data 
dataset = pd.DataFrame()
dataset["overall"] = df["overall"]
dataset["reviewText"] = df["reviewText"]
dataset["summary"] = df["summary"]
dataset.iloc[0:5]

#%%
#Data cleaning and text preprocessing

def cleaning(rawReview):
    #1. Remove non-letters
    letters = re.sub("[^a-zA-Z]", " ", rawReview) 
    
    #2. Convert to lower case
    letters = letters.lower()
    
    #3. Tokenize
    tokens = nltk.word_tokenize(letters)
    
    #4. Convert the stopwords list to "set" data type
    stops = set(nltk.corpus.stopwords.words("english"))
    
    #5. Remove stop words
    words = [w for w in tokens if not w in stops]
    
    #6. Stemming
    words = [nltk.stem.SnowballStemmer("english").stem(w) for w in words] 
    
    #7. Join the words back into one string separated by space, and return the result.
    return " ".join(words)

dataset["cleanReview"] = dataset["reviewText"].apply(cleaning)
dataset["cleanSummary"] = dataset["summary"].apply(cleaning)

#Save processed data set in order to retrieve it
dataset.to_pickle("./Dataset/processedToysAndGames.pkl")


#%%
#Retrieve precedent saved dataset
dataset = pd.read_pickle("./Dataset/processedToysAndGames.pkl")
dataset.head()

#%%
# Data visualization
#WorldCloud
def cloud(data,backgroundcolor = 'white', width = 800, height = 600):
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (10, 5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

cloud(' '.join(dataset["reviewText"]))
cloud(' '.join(dataset["cleanReview"]))

# Distribution
#%%
# We need to split each words in cleaned review and then count the number of each rows of data frame.
dataset["reviewLenght"] = dataset["cleanReview"].apply(lambda x: len(str(x).split())) #Split() a string into a list where each word is a list item
dataset["summaryLenght"] = dataset["cleanSummary"].apply(lambda x: len(str(x).split()))
dataset.head()

#%% 
sns.catplot(x='reviewLenght',kind='count',data=dataset,orient="h", aspect = 1.5, height = 5)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=50))
plt.show()
print("Median of review lenght: ", dataset["reviewLenght"].median())

#%%
sns.catplot(x='summaryLenght',kind='count',data=dataset,orient="h", aspect = 1.5, height = 5)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
plt.show()
print("Median of review lenght: ", dataset["summaryLenght"].median())

#%%
dataset['overall'].value_counts()
sns.catplot(x='overall',kind='count',data=dataset,orient="h")

#%% 
# Split the dataset
#Create validation and training set
xTrain, xTest, yTrain, yTest = train_test_split(dataset["cleanReview"],
                                                 dataset["overall"],
                                                  test_size=0.2, 
                                                   random_state=1)

#%% 
# Feature extraction and Bag Of Words
def dummy(word):
    return word

#Build vectorizer
vect = []
vect.append(CountVectorizer(binary = True, preprocessor = dummy))
vect.append(CountVectorizer(preprocessor = dummy))

#Create bag of words representation for bernoulli model
xTrainBern = vect[0].fit_transform(xTrain) #Learn the vocabulary dictionary and return term-document matrix.
xTestBern = vect[0].transform(xTest) #Transform documents to document-term matrix. We only call transform not fit_transform due to the risk of overfitting.

#Create bag of words representation for binomial model
xTrainBin = vect[1].fit_transform(xTrain)
xTestBin = vect[1].transform(xTest)

print("Number of words in train set vocabulary:", len(vect[0].get_feature_names()), "\n")

#%%
# Shed some light on the bag of words representation returned from the CountVectorizer class
def wm2df(wordMatrix, featureNames):   
    # create an index for each row
    docNames = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wordMatrix)]
    dataFrame = pd.DataFrame(data=wordMatrix.toarray(), index= docNames,
                      columns=featureNames)
    return(dataFrame)

#set of documents
documents = ['The quick brown fox.','The the Jumps over the lazy dog!']
#instantiate the vectorizer object
cvec = CountVectorizer(preprocessor = dummy)
#convert the documents into a document-term matrix
wordMatrix = cvec.fit_transform(documents)
#retrieve the terms found in the corpora
featureNames = cvec.get_feature_names()
#create a dataframe from the matrix
wm2df(wordMatrix, featureNames)

#%% 
#Modelling

#Build models
models = []
models.append(BernoulliNB())
models.append(MultinomialNB())

#Train models
models[0].fit(xTrainBern, yTrain)
models[1].fit(xTrainBin, yTrain)

#Make class predictions
yPredBern = models[0].predict(xTestBern)
yPredBin = models[1].predict(xTestBin)

#%% 
# Analysis 

# Accuracy
print('Bernoulli Accuracy:', metrics.accuracy_score(yTest, yPredBern))
print('Multinomial Accuracy:', metrics.accuracy_score(yTest, yPredBin))


#%%
def analysis(feature, target, maxFeatures = None, minDf = 1,
                             ngramRange = (1,1)):
    #Build vectorizer
    vect = []
    vect.append(CountVectorizer(binary = True, max_features = maxFeatures, min_df = minDf, ngram_range = ngramRange,preprocessor = dummy))
    vect.append(CountVectorizer(max_features = maxFeatures, min_df = minDf, ngram_range = ngramRange, preprocessor = dummy))

    xTrain, xTest, yTrain, yTest = train_test_split(feature, target, test_size=0.2, random_state=1)

    #Create bag of words representation for bernoulli model
    xTrainBern = vect[0].fit_transform(xTrain) #Learn the vocabulary dictionary and return term-document matrix.
    xTestBern = vect[0].transform(xTest) #Transform documents to document-term matrix.

    #Create bag of words representation for binomial model
    xTrainBin = vect[1].fit_transform(xTrain)
    xTestBin = vect[1].transform(xTest)

    print("Number of words in vocabulary:", len(vect[0].get_feature_names()), "\n")
    #Build models
    models = []
    models.append(BernoulliNB())
    models.append(MultinomialNB())
    
    #Train models
    models[0].fit(xTrainBern, yTrain)
    models[1].fit(xTrainBin, yTrain)

    #Make class predictions
    yPredBern = models[0].predict(xTestBern)
    yPredBin = models[1].predict(xTestBin)

    print('Bernoulli Accuracy:', metrics.accuracy_score(yTest, yPredBern))
    print('Multinomial Accuracy:', metrics.accuracy_score(yTest, yPredBin))
    
#%%
analysis(dataset['reviewText'],dataset['overall'])
analysis(dataset['cleanReview'],dataset['overall'], maxFeatures = 50)
analysis(dataset['reviewText'],dataset['overall'], maxFeatures = 50)
analysis(dataset['cleanSummary'],dataset['overall'])
analysis(dataset['summary'],dataset['overall'])
analysis(dataset['cleanSummary'],dataset['overall'], maxFeatures = 50)
analysis(dataset['summary'],dataset['overall'], maxFeatures = 50)

#%% 
# # Sentiment analysis

def sentiment(x):
  if x == 2:
    return np.nan
  if x<2:
    return 0
  if x>2:
    return 1
  
#Add sentiment of each review
dataset["sentiment"] = dataset["overall"].apply(sentiment)

#Delete review with score equal to 2
dataset = dataset.dropna()

print("New dataset look like: \n")
dataset.iloc[:2]

#%%
# 
# Repeat all the tests carried out previously

analysis(dataset['cleanReview'],dataset['sentiment'])
analysis(dataset['reviewText'],dataset['sentiment'])
analysis(dataset['cleanReview'],dataset['sentiment'], maxFeatures = 50)
analysis(dataset['reviewText'],dataset['sentiment'], maxFeatures = 50)
analysis(dataset['summary'],dataset['sentiment'])
analysis(dataset['cleanSummary'],dataset['sentiment'])
analysis(dataset['summary'],dataset['sentiment'], maxFeatures = 50)
analysis(dataset['cleanSummary'],dataset['sentiment'], maxFeatures = 50)

#%%
analysis(dataset['cleanReview'],dataset['overall'], minDf = 2)
analysis(dataset['cleanReview'],dataset['overall'], minDf = 3)
analysis(dataset['cleanReview'],dataset['overall'], ngramRange = (1,3))
analysis(dataset['cleanReview'],dataset['overall'], ngramRange = (1,4))
analysis(dataset['cleanReview'],dataset['overall'], minDf = 2, ngramRange = (1,3))
analysis(dataset['cleanReview'],dataset['overall'], minDf = 2, ngramRange = (1,4))
analysis(dataset['cleanReview'],dataset['overall'], minDf = 2, ngramRange = (1,3), maxFeatures = 50)
analysis(dataset['cleanReview'],dataset['overall'], minDf = 3, ngramRange = (1,3))
analysis(dataset['cleanReview'],dataset['overall'], minDf = 3, ngramRange = (1,4))
analysis(dataset['cleanSummary'],dataset['overall'], minDf = 2)
analysis(dataset['cleanSummary'],dataset['overall'], minDf = 5)
analysis(dataset['cleanSummary'],dataset['overall'],  ngramRange = (1,3))
analysis(dataset['cleanSummary'],dataset['overall'],  ngramRange = (1,4))
analysis(dataset['cleanSummary'],dataset['overall'], minDf = 2, ngramRange = (1,3))
analysis(dataset['cleanSummary'],dataset['overall'], minDf = 2, ngramRange = (1,4))
analysis(dataset['cleanSummary'],dataset['overall'], minDf = 3, ngramRange = (1,3))
analysis(dataset['cleanSummary'],dataset['overall'], minDf = 3, ngramRange = (1,4))

#%%
# Learning curve

def dataPlotPreparation( X, y, maxFeatures = None, minDf = 1,ngramRange = (1,1)):
    #Build vectorizer
    vect = []
    vect.append(CountVectorizer(binary = True, max_features= maxFeatures, min_df = minDf, ngram_range = ngramRange))
    vect.append(CountVectorizer( max_features= maxFeatures, min_df = minDf, ngram_range = ngramRange))

    #Convert in bag of words representation the entire dataset
    XBern = vect[0].fit_transform(X)
    XBin = vect[1].fit_transform(X)
    y = y
    
    print("Number of words in vocabulary :", len(vect[0].get_feature_names()), "\n")
    
    return XBern, XBin, y

#Generate learning curve
def plotLearningCurve(estimator, X, y, trainSizes, title):
    #generate learning curve for bernoulli model
    trainSizes, trainScores, validationScores = learning_curve(estimator = estimator, 
                                                            X = X,
                                                             y = y, 
                                                              train_sizes = trainSizes,
                                                                scoring = 'accuracy',
                                                                 n_jobs=-1,
                                                                  cv = 5)

    #Print scores
    #print('Training scores:\n\n', trainScores)
    #print('\nValidation scores:\n\n', validationScores)

    #Calculate the mean
    trainScoresMean = trainScores.mean(axis = 1)
    validationScoresMean = validationScores.mean(axis = 1)
    #print('Mean training scores\n\n', pd.Series(trainScoresMean, index = trainSizes))
    #print('\nMean validation scores\n\n',pd.Series(validationScoresMean, index = trainSizes))
    trainScoresStd = np.std(trainScores, axis=1)
    validationScoresStd = np.std(validationScores, axis=1)

    #Plot
    plt.style.use('seaborn')
    plt.plot(trainSizes, trainScoresMean, 'o-', label = 'Training accuracy')
    plt.plot(trainSizes, validationScoresMean, 'o-', label = 'Validation accuracy')
    plt.fill_between(trainSizes, trainScoresMean - trainScoresStd,
                     trainScoresMean + trainScoresStd, alpha=0.1,
                     color="b")
    plt.fill_between(trainSizes, validationScoresMean - validationScoresStd,
                     validationScoresMean + validationScoresStd, alpha=0.1, color="g")
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0.3,1.2)

#%%
XBern, XBin, y = dataPlotPreparation( dataset["cleanReview"], dataset["sentiment"], maxFeatures = None)

plotLearningCurve(estimator = models[0], 
                  X = XBern, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 10, dtype = "int"),
                     title="Bernoulli")


#%%
plotLearningCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 10, dtype = "int"),
                     title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( dataset["cleanReview"], dataset["overall"], maxFeatures = None)

plotLearningCurve(estimator = models[0], 
                  X = XBern, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 10, dtype = "int"),
                     title="Bernoulli")

#%%
plotLearningCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 10, dtype = "int"),
                     title="Multinomial")

#%% 
# A comparison between the Bernoulli and the Multinomial model

#Generate validation curve
def plotValidationCurve(estimator, X, y, trainSizes, title):
    #generate learning curve for bernoulli model
    trainSizes, trainScores, validationScores = learning_curve(estimator = estimator, 
                                                            X = X,
                                                             y = y, 
                                                              train_sizes = trainSizes,
                                                                scoring = 'accuracy',
                                                                 n_jobs=-1,
                                                                  cv = 5)

    #Print scores
    #print('Training scores:\n\n', trainScores)
    #print('\nValidation scores:\n\n', validationScores)

    #Calculate the mean
    trainScoresMean = trainScores.mean(axis = 1)
    validationScoresMean = validationScores.mean(axis = 1)
    #print('Mean training scores\n\n', pd.Series(trainScoresMean, index = trainSizes))
    #print('\nMean validation scores\n\n',pd.Series(validationScoresMean, index = trainSizes))
    trainScoresStd = np.std(trainScores, axis=1)
    validationScoresStd = np.std(validationScores, axis=1)

    #Plot
    plt.style.use('seaborn')
    plt.plot(trainSizes, validationScoresMean, 'o-', label = title)

    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title("Comparison", fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0.3,1.2)

#%%
XBern, XBin, y = dataPlotPreparation( dataset["cleanReview"], dataset["sentiment"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")


#%%
XBern, XBin, y = dataPlotPreparation( dataset["cleanReview"], dataset["overall"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")


#%%
XBern, XBin, y = dataPlotPreparation( dataset['cleanReview'],dataset['sentiment'], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")


#%%
XBern, XBin, y = dataPlotPreparation( dataset['cleanReview'],dataset['overall'], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")


#%%
XBern, XBin, y = dataPlotPreparation( dataset["cleanSummary"], dataset["sentiment"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")


#%%
XBern, XBin, y = dataPlotPreparation( dataset["cleanSummary"], dataset["overall"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")


#%%
XBern, XBin, y = dataPlotPreparation( dataset["cleanSummary"], dataset["sentiment"], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")


#%%
XBern, XBin, y = dataPlotPreparation( dataset["cleanSummary"], dataset["overall"], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")

#%% 
# A comparison between the Bernoulli and the Multinomial model - other dataset

#%%
#Load dataset in a panda DataFrame
df = pd.read_json('./Dataset/Digital_Music_5.json',lines=True)

digitalMusicDs = pd.DataFrame()
digitalMusicDs["overall"] = df["overall"]
digitalMusicDs["reviewText"] = df["reviewText"]
digitalMusicDs["summary"] = df["summary"]

#Apply data cleaning and text preprocessing to all dataset
#Add the processed data to the original data. 
digitalMusicDs["cleanReview"] = digitalMusicDs["reviewText"].apply(cleaning)
digitalMusicDs["cleanSummary"] = digitalMusicDs["summary"].apply(cleaning)

#Add sentiment of each review
digitalMusicDs["sentiment"] = digitalMusicDs["overall"].apply(sentiment)

#Review length
digitalMusicDs["reviewLenght"] = digitalMusicDs["cleanReview"].apply(lambda x: len(str(x).split())) #Split() a string into a list where each word is a list item
digitalMusicDs["summaryLenght"] = digitalMusicDs["cleanSummary"].apply(lambda x: len(str(x).split()))

#Add sentiment of each review
digitalMusicDs["sentiment"] = digitalMusicDs["overall"].apply(sentiment)

#Delete review with score equal to 2
digitalMusicDs = digitalMusicDs.dropna()

#Save processed data set in order to retrieve it
digitalMusicDs.to_pickle("./Dataset/processedDigitalMusic.pkl")

#%%
#Load dataset in a panda DataFrame
df = pd.read_json('./Dataset/Grocery_and_Gourmet_Food_5.json',lines=True)

groceryAndGourmetFoodDs = pd.DataFrame()
groceryAndGourmetFoodDs["overall"] = df["overall"]
groceryAndGourmetFoodDs["reviewText"] = df["reviewText"]
groceryAndGourmetFoodDs["summary"] = df["summary"]

#Apply data cleaning and text preprocessing to all dataset
#Add the processed data to the original data. 
groceryAndGourmetFoodDs["cleanReview"] = groceryAndGourmetFoodDs["reviewText"].apply(cleaning)
groceryAndGourmetFoodDs["cleanSummary"] = groceryAndGourmetFoodDs["summary"].apply(cleaning)

#Add sentiment of each review
groceryAndGourmetFoodDs["sentiment"] = groceryAndGourmetFoodDs["overall"].apply(sentiment)

#Review length
groceryAndGourmetFoodDs["reviewLenght"] = groceryAndGourmetFoodDs["cleanReview"].apply(lambda x: len(str(x).split())) #Split() a string into a list where each word is a list item
groceryAndGourmetFoodDs["summaryLenght"] = groceryAndGourmetFoodDs["cleanSummary"].apply(lambda x: len(str(x).split()))

#Add sentiment of each review
groceryAndGourmetFoodDs["sentiment"] = groceryAndGourmetFoodDs["overall"].apply(sentiment)

#Delete review with score equal to 2
groceryAndGourmetFoodDs = groceryAndGourmetFoodDs.dropna()

#Save processed data set in order to retrieve it
groceryAndGourmetFoodDs.to_pickle("./Dataset/processedGroceryAndGourmetFood.pkl")

#%% 
# Some statistics for both dataset
#%%
#Save processed data set in order to retrieve it
digitalMusicDs = pd.read_pickle("./Dataset/processedDigitalMusic.pkl")
print("Dataset dimension: ",digitalMusicDs.shape)
print("Median of review lenght: ", digitalMusicDs["reviewLenght"].median())
print("First two rows of data:")
digitalMusicDs.iloc[:2]

#%%
print("Distribution of overall score:")
sns.catplot(x='overall',kind='count',data=digitalMusicDs,orient="h")


#%%
#Save processed data set in order to retrieve it
groceryAndGourmetFoodDs = pd.read_pickle("./Dataset/processedGroceryAndGourmetFood.pkl")
print("Dimension of data:", groceryAndGourmetFoodDs.shape)
print("Median of review lenght: ", groceryAndGourmetFoodDs["reviewLenght"].median())
print("First two rows of data:")
groceryAndGourmetFoodDs.iloc[:2]

#%%
print("Distribution of overall score:")
sns.catplot(x='overall',kind='count',data=groceryAndGourmetFoodDs,orient="h")

#%% [markdown]
# Test
# We will repeat the test in the same order of the previus chapter

# Digital music dataset
#%%
models = []
models.append(BernoulliNB())
models.append(MultinomialNB())

#%%
XBern, XBin, y = dataPlotPreparation( digitalMusicDs["cleanReview"], digitalMusicDs["sentiment"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( digitalMusicDs["cleanReview"], digitalMusicDs["overall"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( digitalMusicDs["cleanReview"], digitalMusicDs["sentiment"], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( digitalMusicDs["cleanReview"], digitalMusicDs["overall"], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( digitalMusicDs["cleanSummary"], digitalMusicDs["sentiment"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( digitalMusicDs["cleanSummary"], digitalMusicDs["overall"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( digitalMusicDs["cleanSummary"], digitalMusicDs["sentiment"], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( digitalMusicDs["cleanSummary"], digitalMusicDs["overall"], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")

#%% 
# Grocery and gourmet food dataset
XBern, XBin, y = dataPlotPreparation( groceryAndGourmetFoodDs["cleanReview"], groceryAndGourmetFoodDs["sentiment"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( groceryAndGourmetFoodDs["cleanReview"], groceryAndGourmetFoodDs["overall"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( groceryAndGourmetFoodDs["cleanReview"], groceryAndGourmetFoodDs["sentiment"], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( groceryAndGourmetFoodDs["cleanReview"], groceryAndGourmetFoodDs["overall"], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( groceryAndGourmetFoodDs["cleanSummary"], groceryAndGourmetFoodDs["sentiment"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( groceryAndGourmetFoodDs["cleanSummary"], groceryAndGourmetFoodDs["overall"], maxFeatures = None)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( groceryAndGourmetFoodDs["cleanSummary"], groceryAndGourmetFoodDs["sentiment"], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")

#%%
XBern, XBin, y = dataPlotPreparation( groceryAndGourmetFoodDs["cleanSummary"], groceryAndGourmetFoodDs["overall"], maxFeatures = 50)

plotValidationCurve(estimator = models[0], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                     title="Bernoulli")
plotValidationCurve(estimator = models[1], 
                  X = XBin, 
                   y = y, 
                    trainSizes = np.linspace(20, XBern.shape[0]-(XBern.shape[0]*20)/100-1, 20, dtype = "int"),
                    title="Multinomial")




