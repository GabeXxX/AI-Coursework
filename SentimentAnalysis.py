#%%
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from scipy import stats

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
def cloud(data,backgroundcolor = 'white', width = 800, height = 600):
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

cloud(' '.join(dataset["cleanReview"]))
cloud(' '.join(dataset["cleanSummary"]))

#%%
#Distribution
# We need to split each words in cleaned review and then count the number of each rows of data frame.

dataset["wordFrequency"] = dataset["cleanReview"].apply(lambda x: len(str(x).split())) #Split() a string into a list where each word is a list item
dataset["uniqueWordFrequency"] = dataset["cleanReview"].apply(lambda x: len(set(str(x).split())))

dataset["wordFrequencySummary"] = dataset["cleanSummary"].apply(lambda x: len(str(x).split()))
dataset["uniqueWordFrequencySummary"] = dataset["cleanSummary"].apply(lambda x: len(set(str(x).split())))

fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(10,5)

sns.distplot(dataset["wordFrequency"], bins = 90, ax=axes[0], fit = stats.norm)
(mu0, sigma0) = stats.norm.fit(dataset["wordFrequency"])
axes[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
axes[0].set_title("Distribution Word Frequency")
axes[0].axvline(dataset["wordFrequency"].median(), linestyle='dashed')

print("median of word frequency: ", dataset["wordFrequency"].median())

sns.distplot(dataset["uniqueWordFrequency"], bins = 90, ax=axes[1], color = 'r', fit = stats.norm)
(mu1, sigma1) = stats.norm.fit(dataset["uniqueWordFrequency"])
axes[1].set_title("Distribution Unique Word Frequency")
axes[1].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1)],loc='best')
axes[1].axvline(dataset["uniqueWordFrequency"].median(), linestyle='dashed')
print("median of uniuqe word frequency: ", dataset["uniqueWordFrequency"].median())

#%%
dataset.iloc[:2]

#%%
