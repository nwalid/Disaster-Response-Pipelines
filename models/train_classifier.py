# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier



import warnings

warnings.simplefilter('ignore')


def load_data(database_filepath):
    """
    Load data from SQLite database and split into features and target
    """

    # load data
    engine = create_engine('sqlite:///' +database_filepath)
    df = pd.read_sql_table(database_filepath, con=engine)

    # Define feature and target variables X and Y
    X = df['message']
    Y = df.iloc[:,4:]
    categories = Y.columns.values 

    return X, Y, categories

def tokenize(text):

    """
    Convert to lower case , remover special characters and lemmatize texts
    """
      
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize words
    tokens = word_tokenize(text)
      
    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return stemmed

def build_model():

    """
    Build machine learning model with pipeline
    """

    # create pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters =  {'clf__estimator__n_estimators': [50,100],
            
                'clf__estimator__max_depth' : [2,3,4]}

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model,X_test,Y_test,category_names):

    '''
    INPUT 
        pipeline: The model that is to be evaluated
        X_test: Input features, testing set
        y_test: Label features, testing set
        category_names: List of the categories
    OUTPUT
        Display precision, recall, f1-score of model scored on testing set
    '''
    # print scores
 
    # make predictions with model
    y_pred = model.predict(X_test)
     

    for i in range(0, len(category_names)):
        print(classification_report(Y_test[category_names[i]],y_pred[:, i]))



def save_model(model, model_filepath):

    """
    Pickle model to designated file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
