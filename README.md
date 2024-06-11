# ALS_Movie_Recommendation_Algorithm
An Alternating Least Squares (ALS) algorithm trained off of GroupLen's MovieLens ml-latest-small dataset. Applies collaborative filtering by approximating a User-Movie Matrix used to recommend a movie for a given user.

## main.py
The main execution block of the script main.py is responsible for loading the data, creating a user-item matrix in CSR format, partitioning the data into training and testing sets, initializing and training an ALS model, and visualizing the training and testing losses. Below is a step-by-step description of what happens in this block:

### Argument Handling:
- The script expects two command-line arguments: the file path to the sparse matrix and the file path to the index-class mapping.
- If the required arguments are not provided, the script prints a usage message and exits.

### Data Loading:
- The user-item interaction data is loaded from a CSV file using the load_data function. This function reads the file, drops unnecessary columns, and shuffles the data.

### CSR Matrix Creation:
- The script calculates the number of unique users and items, then constructs a Compressed Sparse Row (CSR) matrix from the user-item data using sparse.csr_array.

### Data Partitioning:
- The data is partitioned into training and testing sets with a 70/30 split.

### Multi-Indexing:
- Both the training and testing data are multi-indexed by userId and movieId to facilitate efficient lookups during model training and evaluation.

### Model Initialization:
- An ALS model is initialized using the ALT_ALS_Model class. The model parameters include the number of latent factors, learning rate, number of iterations, convergence threshold, and regularization factors.

### Model Training:
- The ALS model is trained on the training data, with the training and testing losses recorded at each iteration.

### Loss Visualization:
- The training and testing losses are plotted over the iterations to visualize the model's performance during training.

### Loading Index-Item Mapping:
- The index-to-item mapping is loaded from a CSV file using the load_index_item function. This mapping is used to provide readable item recommendations.

### Recommendations:
- The trained model is used to recommend 5 movies for a specific user (user 2 in this case), and the recommendations are printed out.

---

## alternate_training.py

The script alternate_training.py defines the ALT_ALS_Model class, which implements the Alternating Least Squares (ALS) algorithm for collaborative filtering. Below is a detailed description of the class and its methods, focusing on how it is utilized in the main.py script.

### Class: ALT_ALS_Model

The ALT_ALS_Model class is designed to train a collaborative filtering model using the ALS algorithm. The main components of this class include initialization, random array generation, dot product computation, error calculation, gradient computation, and model training.

### Key Methods and Attributes:

#### `__init__`: 
- Initializes the ALS model with specified parameters, including the number of users, items, latent factors, learning rate, maximum iterations, convergence threshold, and regularization factors. It also initializes user and item factors and biases.

#### `randomize_narray`: 
- Generates a randomized array for initializing factors and biases based on a Gaussian distribution.

#### `dot_product`: 
- Computes the dot product between user and item vectors, adding biases to generate the predicted rating.

#### `means_squared_error`: 
- Calculates the Mean Squared Error (MSE) for the predicted ratings against the actual ratings in the training data.

#### `compute_user_gradients` and `compute_item_gradients`: 
- Compute gradients for user and item factors and biases, which are used to update these parameters during training.

#### `train`: 
- Implements the ALS training loop, alternating between updating user and item factors and biases, and recording the training and testing losses at each iteration.

#### `recommend_nitems`: 
- Recommends a specified number of items for a given user based on the trained model.
