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
    return index_item_dict

def to_csr(dataframe:pd.DataFrame, user_name:str, item_name:str, weight_name:str):
    """
    Returns a SciPy CSR_Matrix from a given pandas dataframe representing the 
    user-item matrix/dataset
    """
    users = dataframe[user_name].max()
    items = dataframe[item_name].max()
    coo = sparse.coo_matrix((dataframe[weight_name].astype(float), (dataframe[user_name], dataframe[item_name])), shape=(users+1,items+1))
    return coo.tocsr()

def output_csr_as_txt(filename: str, csr: sparse._csr.csr_matrix):
    file = open(filename,'w')
    for i in range(csr.shape[0]):
        for j in csr[i].nonzero()[1]:
            file.write(str(i)+' ' +str(j)+' '+str(csr[i,j])+'\n')
    file.close
    return



if __name__=="__main__":
    #Args: python3 main.py <sparse-matrix> <index-class>
    #Maybe add userId, movieId, rating as arguments?
    if len(sys.argv) != 3:
        print("Args need to be: python3 main.py <sparse-matrix> <index-class>")
        sys.exit(-1)
    #Load datasets into dataframes
    print("\tLoading movie/user matrix:", sys.argv[1], "\n\tUsing", sys.argv[2], "as key matrix.")
    train_user_items,test_user_items = load_user_item(sys.argv[1], "userId", "movieId", "rating")
    index_item_dict = load_index_item(sys.argv[2], "movieId", "title")
    #Load user-item dataframe into csr_matrix format
    #TODO: Get rid of extra row in csr_matrix? Or ignore?
    train_user_items = to_csr(train_user_items, "userId", "movieId", "rating")
    output_csr_as_txt("user-movie-rating.txt", train_user_items)
    print("successful!")

