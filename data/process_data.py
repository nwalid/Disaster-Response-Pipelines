# import libraries
import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge (messages,categories, on='id')


    return df


def clean_data(df):
    # split categories into columns
    categories = df.categories.str.split(pat=';',expand=True)

    # rename columns
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1

    for column in categories:
    # set each value to be the last character of the string
        categories[column]= categories[column].astype(str).str[-1]
    
    # convert column from string to numeric
        categories[column]= categories[column].astype(int)

    #Replace categories column in df with new category columns.
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    print(df) 
    #Remove duplicates
    df = df.drop_duplicates( inplace=False)
    print(df)
    return df


def save_data(df, database_filename):

    print(database_filename)
    conn_string = 'sqlite:///' + database_filename
    print(conn_string)
    engine = create_engine(conn_string)
    #engine = create_engine('sqlite:///data//database_filename')
    df.to_sql('DisasterMessages', engine,if_exists='replace', index=False)
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
