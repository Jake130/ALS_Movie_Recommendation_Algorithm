import csv
import os
import sys
import pandas as pd
from scipy import sparse
import implicit
from fastai.collab import *
from fastai.tabular.all import *
from collections import Counter
import torch
import torch.nn as nn

# Set OpenBLAS to use only one thread
os.environ['OPENBLAS_NUM_THREADS'] = '1'

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

def output_csr_as_txt(filename: str, csr: sparse._csr.csr_matrix, column_names: list):
    """
    Writes the contents of a CSR (Compressed Sparse Row) matrix to a text file, including column headers.
    
    Parameters:
    filename (str): The name of the output text file.
    csr (sparse._csr.csr_matrix): The CSR matrix containing the user-item data.
    column_names (list): A list of column names to include as headers in the output file.
    """
    file = open(filename,'w')
    # Write the header with column names and types
    header = ' '.join(column_names)
    file.write(header + '\n')
    # Write data
    for i in range(csr.shape[0]):
        for j in csr[i].nonzero()[1]:
            file.write(f"{i} {j} {csr[i, j]}\n")
    file.close
    
def numpy_array_to_txt(filename: str, array: np.ndarray):
    """
    Writes a 2D NumPy array to a text file, where each row represents a user or item and their factors.
    
    Parameters:
    filename (str): The name of the output text file.
    array (np.ndarray): The 2D NumPy array to write to the file.
    """
    file = open(filename, 'w')
    for row in array:
        row_str = ' '.join(map(str, row))
        file.write(row_str + '\n')
    file.close

def create_als_model() -> implicit.als.AlternatingLeastSquares:
    """Returns model trained of of implementing ALS on the 
    user-items csr_matrix"""
    return implicit.als.AlternatingLeastSquares(factors=50, iterations=10, regularization=0.01)

def accuracy_tester(train_data, test_data, model) -> float:
    """
    Evaluates the accuracy of a trained ALS model by computing the Root Mean Squared Error (RMSE) on the test dataset.

    Parameters:
    train_data (pd.DataFrame): The training dataset containing user-item interactions.
    test_data (pd.DataFrame): The test dataset containing user-item interactions for evaluation.
    model (implicit.als.AlternatingLeastSquares): The trained ALS model.

    Returns:
    float: The RMSE of the model's predictions on the test dataset.

    Description:
    This function first converts the test dataset into PyTorch tensors for users, items, and ratings. It then extracts the
    latent factors (user and item embeddings) from the trained ALS model and converts these factors into PyTorch tensors.
    Using these factors, it predicts the ratings for the user-item pairs in the test dataset. The Mean Squared Error (MSE)
    between the predicted ratings and the true ratings is calculated, and the square root of this MSE gives the RMSE. This
    RMSE value is returned as a measure of the model's accuracy.

    Example:
    >>> train_data, test_data = load_user_item('ratings.csv', 'userId', 'movieId', 'rating')
    >>> model = create_als_model()
    >>> model.fit(train_user_items_csr)
    >>> rmse = accuracy_tester(train_data, test_data, model)
    >>> print(f'RMSE: {rmse}')
    """
    # Convert test data to PyTorch tensors
    user_ids = torch.tensor(test_data['userId'].values - 1)  # Adjusting for 0-based indexing
    item_ids = torch.tensor(test_data['movieId'].values - 1)  # Adjusting for 0-based indexing
    true_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float32)
    
    # Get latent factors from the model
    user_factors = model.user_factors
    item_factors = model.item_factors

    # Convert latent factors to PyTorch tensors
    user_factors_tensor = torch.tensor(user_factors, dtype=torch.float32)
    item_factors_tensor = torch.tensor(item_factors, dtype=torch.float32)

    # Define the MSE Loss function
    mse_loss = nn.MSELoss()

    # Predict ratings
    predicted_ratings = (user_factors_tensor[user_ids] * item_factors_tensor[item_ids]).sum(dim=1)

    # Calculate MSE loss
    loss = mse_loss(predicted_ratings, true_ratings)

    # Calculate RMSE
    rmse = torch.sqrt(loss).item()
    print(f'RMSE: {rmse}')

    return rmse

if __name__=="__main__":
    #Args: python3 main.py <sparse-matrix> <index-class>
    #Maybe add userId, movieId, rating as arguments?
    if len(sys.argv) != 3:
        print("Usage: python3 main.py <sparse-matrix> <index-class>")
        sys.exit(-1)
    #Load datasets into dataframes
    print("\tLoading movie/user matrix:", sys.argv[1], "\n\tUsing", sys.argv[2], "as key matrix.")
    train_user_items,test_user_items = load_user_item(sys.argv[1], "userId", "movieId", "rating")
    index_item_dict = load_index_item(sys.argv[2], "movieId", "title")
    merged = train_user_items.merge(index_item_dict, left_on="movieId", right_index=True)


    # will maybe comment back in later
    # began messing with pytorch
    # check this link https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive
    # dls = CollabDataLoaders.from_df(merged, item_name='title', bs=64)
    # dls.show_batch()
    # n_users = len(dls.classes['userId'])
    # n_movies = len(dls.classes['title'])
    # n_factors = 5
    # # Initialize user and item factors
    # # This is part of the process of decomposing the user-item matrix into lower dimensional space
    # user_factors = np.random.rand(n_users, n_factors)
    # item_factors = np.random.rand(n_movies, n_factors)
    
    # print("Initial User Factors:")
    # print(user_factors)
    # print("Initial Item Factors:")
    # print(item_factors)

    # Save the merged DataFrame to a new CSV file
    # merged.to_csv('merged_output.csv', index=False)
    
    #Load user-item dataframe into csr_matrix format
    #TODO: Get rid of extra row in csr_matrix? Or ignore?
    train_user_items_csr = to_csr(train_user_items, "userId", "movieId", "rating")

    #Create Model
    als_model = create_als_model()
    als_model.fit(train_user_items_csr)
    #Get top 5
    n_users = train_user_items["userId"].nunique()
    userids = np.arange(n_users)
    recommendations = []
    for userid in userids:
        movie_ids, ratings = als_model.recommend(userid, train_user_items_csr[userid], N=5)
        # Debugging: Print the movie_ids to see what we get
        print(f"User {(userid + 1)} recommended movie IDs: {movie_ids + 1}")
        # Check if movie_id exists in index_item_dict to avoid KeyError
        valid_movie_ids = [movie_id + 1 for movie_id in movie_ids if movie_id + 1 in index_item_dict.index]
        recommendations.extend(valid_movie_ids)  # Collect all valid recommendations
    
    # Count the occurrences of each recommended movie and get the top 5 most common recommendations
    movie_counter = Counter(recommendations)
    top_5_recommendations = movie_counter.most_common(5)

    print("Top 5 most common recommendations:")
    for movie_id, count in top_5_recommendations:
        movie_title = index_item_dict.loc[movie_id, "title"]
        print(f"{movie_title}: {count} times recommended")

    output_csr_as_txt("user-movie-rating.txt", train_user_items_csr, ["userId", "movieId", "rating"])
    # numpy_array_to_txt("user-matrix.txt", user_factors)
    # numpy_array_to_txt("item-matrix.txt", item_factors)

    rmse = accuracy_tester(train_user_items, test_user_items, als_model)

    # RMSE (Root Mean Squared Error) is a standard way to measure the error of a model in predicting 
    # quantitative data. It provides an aggregate measure of the accuracy of the model's predictions. 
    # The lower the RMSE, the better the model's performance in terms of predictive accuracy.
    print(f"RMSE: {rmse}")

    # I would be weary of the below print statement, I was working on the above numpy to txt function
    # and I accidentally outputted a 1.3 GB txt file. It DOES NOT do this anymore (I had an extra for loop by accident), 
    # but considering that this is small data, we might want to consider this in terms of efficiency, 
    # storage, and timing for the bigger data set.
    print("Done!")