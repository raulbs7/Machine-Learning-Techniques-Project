# Machine Learning Techniques Projects
This is a repository in which we are going to upload our differents projects of machine learning.

## Dengue Unsupervised Learning Project

The purpose of this project is to discover some patterns of unlabeled data. The data used for this investigation is a data source from the web page of Driven Data, which belongs to a competition of predicting disease spread.

This data has been preprocessed and normalized. The goal was treat this data with unsupervised machine learning techniques, eliminating some features by dimensionality reduction and, after that, executing on them clustering techniques.

## NLP Supervised Learning Project 

This project consists on using Natural Language Processing techniques to analyze a bunch of tweets with offensive language and hatred speech.

1. [ File descriptions ](#desc)
2. [ Technologies used ](#usage)
3. [ Structure ](#structure)


<a name="desc"></a>
### **File descriptions**

* [1_Preprocessing.ipynb](NLP_Supervised_Project/1_Preprocessing.ipynb): notebook in which is done the preprocessing of the tweets, removing all the unnecessary data like links, some symbols, emojis and, also, tokenize the tweets for the lemmatization of them.
* [2_Vectorization.ipynb](NLP_Supervised_Project/2_Vectorization.ipynb): this notebook is in charge of taking the preprocessed tweets and vectorize them with techniques as TFIDF, that express the relevance of the words appearing in each tweet, or sentiment analysis, which analyzes the type of speech.
* [3_Feature_Selection.ipynb](NLP_Supervised_Project/3_Feature_Selection.ipynb): this notebook is made for the selection of the best features obtained by the vectorization, with SelectKBest tool.
* [4_Classification.ipynb](NLP_Supervised_Project/4_Classification.ipynb): finally, the purpose of this last notebook is to make two models of classifation of the tweets. This models classifate them betweent hate speech tweets, offensive language tweets and neither.
* [labeled_data.csv](NLP_Supervised_Project/res/labeled_data.csv): this CSV file is the one that contains the raw data to be processed and evaluated to create our models.
* [emo_unicode.py](NLP_Supervised_Project/res/emo_unicode.py): this is a python module that contains dictionaries with emojis and literal meanings.
* [big.txt](NLP_Supervised_Project/res/big.txt): a large text file used for the corrections of words.
* [processed_tweets.csv](NLP_Supervised_Project/res/processed_tweets.csv): it is the CSV file obtained by the preprocessing of the tweets.
* [vectorized_tweets.npz](NLP_Supervised_Project/res/vectorized_tweets.npz): it is an special format that contains a sparse matrix obtained by the vectorization of the preprocessed tweets.
* [selected_features.npz](NLP_Supervised_Project/res/selected_features.npz): they are the features selected as the best of all the features obtained by the vectorization.

<a name="usage"></a>
### **Technologies used**

* [scikit-learn](https://scikit-learn.org/stable/index.html)
* [seaborn](https://seaborn.pydata.org)
* [matplotlib](https://matplotlib.org)
* [scipy](https://www.scipy.org)
* [pandas](https://pandas.pydata.org)
* [numpy](https://numpy.org)

<a name="structure"></a>
### **Structure**

1. NLP PREPROCESSING
    * Imports
    * Preprocessing steps 
        * Removing mentions
        * Removing URLs
        * Removal of capital letters
        * Removal of contractions
        * Removing symbols
        * Removal of repeated words
        * Removal of numbers
        * Tokenization
        * Stopwords
        * Conversion of emojis
        * Correction of wrong words
        * Lemmatization
        * Prepare dataframe
2. VECTORIZATION
    * Imports
    * Importing dataset
    * TFIDF
    * TFIDF with N-grams
    * TFIDF with N-grams and POS-tagging
    * TFIDF with N-grams, POS-tagging and other features
        * Number of RT's
        * Sentiment Analysis
        * Hatred N-gram dictionary
    * Unify all configurations
3. FEATURE SELECTION
    * Imports
    * Importing dataset
    * Selection of the best features
    * Exporting data
4. TWEETS CLASSIFICATION
    * Imports
    * Importing files
    * Preprocessing
    * Naive Bayes Classification
        * Splitting train and test
        * Cross validation
        * Building model
        * Evaluation of the model
    * K-Nearest-Neighbours
        * Splitting train and test
        * Cross validation

## Dengue Supervised Learning Project 

For this project, the goal will be determine a model, to predict the spread of this disease in two cities, with supervised learning techniques. The data used for this competition is the same as the project of Dengue Unsupervised Learning Project from Data Driven. In this investigation the minimization of the error will be the objective.

1. [ File descriptions ](#desc)
2. [ Technologies used ](#usage)
3. [ Structure ](#structure)


<a name="desc"></a>
### **File descriptions**

<a name="usage"></a>
### **Technologies used**

* [scikit-learn](https://scikit-learn.org/stable/index.html)
* [seaborn](https://seaborn.pydata.org)
* [matplotlib](https://matplotlib.org)
* [scipy](https://www.scipy.org)
* [pandas](https://pandas.pydata.org)
* [numpy](https://numpy.org)

<a name="structure"></a>
### **Structure**

## Autores ✒️

* **Raúl Bernalte Sánchez** - [raulbs7](https://github.com/raulbs7)
* **Elena María Ruiz**  - [elenamariaruiz](https://github.com/elenamariaruiz)