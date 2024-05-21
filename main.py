import csv
import sys
import pandas as pd
from scipy import sparse



#Read ratings.csv into a train/test pandas dataframe
def load_user_item(filepath:str, user_name:str, item_name:str, weight_name:str):
    """Read user-item sparse matrix into csr format"""
    user_items = pd.read_csv(filepath, sep=',')
    for column in user_items.columns:
        if column!=user_name and column!=item_name and column!=weight_name:
            user_items.drop(columns=column, inplace=True)
    #shuffle data
    user_items = user_items.sample(frac=1).reset_index(drop=True)
    #partition data 70/30
    partition_point = (user_items.shape[0] * 7)//10
    test_user_items = user_items.iloc[partition_point:]
    return (user_items.iloc[:partition_point], test_user_items)

#Read index-movies into pandas dataframe
#Parameters are INDEX_NAME and VALUE_NAME to indicate columns to extract
def load_index_item(filepath:str, index_name:str, value_name:str):
    """
    Read index-item into pandas dataframe
    Parameters are INDEX_NAME and VALUE_NAME to indicate columns to extract
    """
    index_item_dict = pd.read_csv(filepath, sep=',')
    index_item_dict.set_index(index_name, inplace=True)
    for column in index_item_dict.columns:
        #remove all unecessary columns
        if column!=index_name and column!=value_name:
            index_item_dict.drop(columns=[column], axis=1, inplace=True)
    #print(index_item_dict.head(10))
    return index_item_dict

def to_csr(dataframe:pd.DataFrame, user_name:str, item_name:str, weight_name:str):
    users = dataframe[user_name].max()
    items = dataframe[item_name].max()
    coo = sparse.coo_matrix((dataframe[weight_name].astype(float), (dataframe[user_name], dataframe[item_name])), shape=(users+1,items+1))
    return coo.tocsr()




if __name__=="__main__":
    #Args: python3 main.py <sparse-matrix> <index-class>
    if len(sys.argv) != 3:
        print("Args need to be: python3 main.py <sparse-matrix> <index-class>")
        sys.exit(-1)
    #load index-value
    train_user_items,test_user_items = load_user_item(sys.argv[1], "userId", "movieId", "rating")
    index_item_dict = load_index_item(sys.argv[2], "movieId", "title")
    print(train_user_items.shape[0])
    print(test_user_items.shape[0])
    #Get rid of extra row
    train_user_items = to_csr(train_user_items, "userId", "movieId", "rating")
    #print(user_items.loc[1, "title"])
    #train_data = user_items.sample(n=int(0.7*len(user_items)), axis=0)
    #print(train_data.head(10))
    #test_data = user_items.sample(n=0.3*len(user_items))
