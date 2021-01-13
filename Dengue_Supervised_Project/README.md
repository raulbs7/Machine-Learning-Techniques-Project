# Dengue Supervised Learning Project 

For this project, the goal will be determine a model, to predict the spread of this disease in two cities, with supervised learning techniques. The data used for this competition is the same as the project of Dengue Unsupervised Learning Project from Data Driven. In this investigation the minimization of the error will be the objective.

1. [ File descriptions ](#desc)
2. [ Technologies used ](#usage)
3. [ Structure ](#structure)


<a name="desc"></a>
## **File descriptions**

* [1_Preprocessing.ipynb](1_Preprocessing.ipynb): this notebook is in charge of preprocessing data, in order to be ready for the training.
* [2_Baseline.ipynb](2_Baseline.ipynb): this notebook describes the process of trying
* [3_Optimization_Model.ipynb](3_Optimization_Model.ipynb): notebook in which it will be done the lines of improvements and, also, the optimization of the choosen model determined by the baseline.
* [dengue_features_test.csv](res/dengue_features_test.csv): this is the original dataset for testing.
* [dengue_features_train.csv](res/dengue_features_train.csv): this is the original dataset for training.
* [processed_test.csv](res/processed_test.csv): this is the preprocessed data for testing.
* [processed_train.csv](res/processed_train.csv): this is the preprocessed data for training.
* [submission_baseline.csv](res/submission_baseline.csv): this data are the predictions obtained by the model created in the baseline.
* [submission_optimization.csv](res/submission_optimization.csv): this data are the predictions obtained by the model created by the optimized model.

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

1. PREPROCESSING
    * Initialization
    * Removal of redundant and unnecesary features 
    * Variable categorization
    * Discretization
    * Elimination of outliers
    * Null values treatment
    * Saving data
2. BASELINE
    * Initialization
    * Analysis of data
    * Models
        * Random Forests
        * Linear Regression
        * Gradient Boosting
        * K-nearest-neighbors
    * First submission
3. OPTIMIZATION MODEL
    * Initialization
    * Importing dataset
    * Selection of the best features
        * San Juan model
        * Iquitos model
        * Submission of the two models


## Autores ✒️

* **Raúl Bernalte Sánchez** - [raulbs7](https://github.com/raulbs7)
* **Elena María Ruiz**  - [elenamariaruiz](https://github.com/elenamariaruiz)
