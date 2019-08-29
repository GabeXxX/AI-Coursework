import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import learning_curve
from sklearn import metrics

from wordcloud import WordCloud, STOPWORDS

#Create a dataframe from a word matrix return by CountVectorizer
def wm2df(wordMatrix, featureNames):   
    # create an index for each row
    docNames = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wordMatrix)]
    dataFrame = pd.DataFrame(data=wordMatrix.toarray(), index= docNames,
                      columns=featureNames)
    return(dataFrame)

#Plot learning curves of a machine learning alghoritm
def plotLearningCurve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), scoring = 'accuracy',):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Train set size")
    plt.ylabel("Accuracy")

    #generate evaluation curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    
    #calculate mean value for plotting
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)


    #plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    #plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    plt.grid()

    return plt

#Calculate accuracy, precision, recall, and F-measure of class predictions
def evalPredictions(ytest, ypred):
    print('accuracy:', metrics.accuracy_score(ytest, ypred))
    print('precision:', metrics.precision_score(ytest, ypred, average='weighted'))
    print('recall:', metrics.recall_score(ytest, ypred, average='weighted'))
    print('F-measure:', metrics.f1_score(ytest, ypred, average='weighted'))

#WorldCloud
def cloud(data,backgroundcolor = 'white', width = 800, height = 600):
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

#Apply data cleaning and text preprocessing to all dataset
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