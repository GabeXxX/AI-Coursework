# AmazonSentimentAnalysis

In the following project the Naive Bayes probabilistic method is used to classify product reviews on amazon.
Specifically, a comparison was made between the Bernoulli version and the Multinomial version using three datasets available at http://jmcauley.ucsd.edu/data/amazon/.

Before performing the tests, some text analysis techniques were investigated. In particular:

1. Some text cleaning techniques was studied.
2. We tried to perform elementary operations on the visualization of data.
3. The bag of words model was studied.

The Scikit-learn library was used to implement both Naive Bayes methods.
We have tried to describe every piece of code written in the project, as well as tried to describe the functioning of the methods taken from the Scikit library.

The theory behind Naive Bayes has not been described. However, some techniques used in machine learning have been briefly described becouse adopted by the Scikit library.

Various tests were carried out. In particular, in chronological order:

1. We tried to classify the reviews trying to predict the overall score of the review (from 1 to 5).

2. We tried to classify the reviews trying to predict only their positvity or negativity  (or 1 or 0).

3. Some tests were carried out to try to improve the accuracy of the predicton of the overall score (from 1 to 5).

4. First conclusions were drawn from these tests.

5. The learning curves of the two methods were plotted and compared, using the same tests carried out previously.

6. Conclusions were drawn by observing these learning curves, generalizing the conclusions given previously.

The complete project is included in the Jupiter Notebook NaiveBayesProject.ipynb file.

A more compact version was also made available including only the python code used in the project, which however does not include any explanation about the code and which may be more difficult to read without first reading the previous file.

Finally, a pdf conversion of the project is also available.
This may not be visually satisfying but it is useful if you have problems viewing the Jupiter notebook version.

The report summerize and analyze the result of the tests performed.
