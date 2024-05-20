import csv
import sys
import pandas as pd
from scipy import sparse



#Read ratings.csv into a csr_matrix
def load_user_item(filepath:str, user_name:str, item_name:str, weight_name:str):
    """Read user-item sparse matrix into csr format"""
    user_items = pd.read_csv(filepath, sep=',')
    user_items.set_index([user_name, item_name], inplace=True)
    for column in user_items.columns:
        if column!=user_name and column!=item_name and column!=weight_name:
            user_items.pop(column)
    print(user_items.head(10))
    return user_items



#Read index-movies into pandas dataframe
#Parameters are INDEX_NAME and VALUE_NAME to indicate columns to extract
def load_index_item(filepath:str, index_name:str, value_name:str):
    """
    Read index-item into pandas dataframe
    Parameters are INDEX_NAME and VALUE_NAME to indicate columns to extract
    """
    index_item_dict = pd.read_csv(filepath, sep=',')
    index_item_dict.set_index(index_name)
    for column in index_item_dict.columns:
        #remove all unecessary columns
        if column!=index_name and column!=value_name:
            index_item_dict.drop(columns=[column], axis=1)
    print(index_item_dict.head(10))
    return index_item_dict




if __name__=="__main__":
    #Args: python3 main.py <sparse-matrix> <index-class>
    if len(sys.argv) != 3:
        print("Args need to be: python3 main.py <sparse-matrix> <index-class>")
        sys.exit(-1)
    #load index-value
    index_item_dict = load_index_item(sys.argv[2], "movieId", "title")
    user_items = load_user_item(sys.argv[1], "userId", "movieId", "rating")
    print(user_items.loc[1, "title"])
    #train_data = user_items.sample(n=int(0.7*len(user_items)), axis=0)
    #print(train_data.head(10))
    #test_data = user_items.sample(n=0.3*len(user_items))
