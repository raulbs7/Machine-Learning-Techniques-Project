# NLP Supervised Project 

This project consists on using Natural Lenguage Processing techniques to analyze a bunch of tweets with offensive language and hatred speech.

1. [ File descriptions ](#desc)
2. [ Technologies used ](#usage)
3. [ Structure ](#structure)

<a name="desc"></a>
## **File descriptions**

* [1_Preprocessing.ipynb](1_Preprocessing.ipynb): notebook in which is done the preprocessing of the tweets, removing all the unnecessary data like links, some symbols, emojis and, also, tokenize the tweets for the lemmatization of them.
* [2_Vectorization.ipynb](2_Vectorization.ipynb): this notebook is in charge of taking the preprocessed tweets and vectorize them with techniques as TFIDF, that express the relevance of the words appearing in each tweet, or sentiment analysis, which analyzes the type of speech.
* [3_Feature_Selection.ipynb](3_Feature_Selection.ipynb): this notebook is made for the selection of the best features obtained by the vectorization, with SelectKBest tool.
* [4_Classification.ipynb](4_Classification.ipynb): finally, the purpose of this last notebook is to make two models of classifation of the tweets. This models classifate them betweent hate speech tweets, offensive language tweets and neither.
* [labeled_data.csv](res/labeled_data.csv): this CSV file is the one that contains the raw data to be processed and evaluated to create our models.
* [emo_unicode.py](res/emo_unicode.py): this is a python module that contains dictionaries with emojis and literal meanings.
* [big.txt](res/big.txt): a large text file used for the corrections of words.
* [processed_tweets.csv](res/processed_tweets.csv): it is the CSV file obtained by the preprocessing of the tweets.
* [vectorized_tweets.npz](res/vectorized_tweets.npz): it is an special format that contains a sparse matrix obtained by the vectorization of the preprocessed tweets.
* [selected_features.npz](res/selected_features.npz): they are the features selected as the best of all the features obtained by the vectorization.

<a name="usage"></a>
## **Technologies used**

* [scikit-learn](https://scikit-learn.org/stable/index.html)
* [seaborn](https://seaborn.pydata.org)
* [matplotlib](https://matplotlib.org)
* [scipy](https://www.scipy.org)
* [pandas](https://pandas.pydata.org)
* [numpy](https://numpy.org)

<a name="structure"></a>
## **Structure**

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

## Autores ✒️

* **Raúl Bernalte Sánchez** - [raulbs7](https://github.com/raulbs7)
* **Elena María Ruiz**  - [elenamariaruiz](https://github.com/elenamariaruiz)