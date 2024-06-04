import pandas as pd
import random
from math import log, exp, sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



class ALS_Model():
    def __init__(self, num_users, num_items, n_latency_factors, eta, max_iters, convergence_threshold, l2_factor, factor_scale, bias_scale):
        """
        Usage:

        model = ALS_Model(<number of users>, <number of items>, <number of latent factors>, <learning rate>, <maximum iterations>, <convergence threshold>, <l2 lambda value>, <scalar for factor initialization>, <scalar for bias initialization>)
        """
        
        # random.seed(42) # you can seed specific randomization to check how specific initialization will impact loss!
        # Try changing the seed to any number and take note of the initial values and how they impact MSE calculation
        self.factor_scale = factor_scale # an adjustable scalar for factor initialization
        self.bias_scale = bias_scale # an adjustable scalar for bias initialization

        self.num_users = num_users
        self.num_items = num_items

        self.user_factors = self.randomize_narray(num_users, n_latency_factors, factor_scale)                  #Randomly Initialize
        self.user_biases = self.randomize_narray(1, num_users, bias_scale)                   #Randomly Initialize
        self.item_factors = self.randomize_narray(n_latency_factors, num_items, factor_scale)                 #Randomly Initialize
        self.item_biases = self.randomize_narray(1, num_items, bias_scale)                  #Randomly initialize
        self.n_latency_factors = n_latency_factors
        self.eta = eta
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.l2_factor = l2_factor

    def randomize_narray(self, M:int, N:int, scale):
        empty = []
        if M<=1:
            for el in range(N):
                empty.append(random.gauss(0.0, scale))
        else:
            for row in range(M):
                new = []
                for el in range(N):
                    new.append(random.gauss(0.0, scale))
                empty.append(new)
        return empty

    def dot_product(self, user_id:int, item_id:int) -> float:
        """Computes dot product between user row and item column specified
        by indices 'user_id' and 'item_id' """
        #This is going to be 0-indexed rather than the 1-index from user/item ids
        result = 0.0
        for i in range(self.n_latency_factors):
            result += self.user_factors[user_id-1][i]*self.item_factors[i][item_id-1]
        result += self.user_biases[user_id-1] + self.item_biases[user_id-1]
        return result

    def sigmoid_range(self, value:int, range=5.0):
        """Squishes value within a sigmoid that has a range of 'range' 
        +/- 2.197 values reults in saturation
        """
        return range/(1+exp(-value))

    def means_squared_error(self, training_data:pd.DataFrame, ratings):
        """Compute MSE over all parameters & the actual values"""
        sum = 0.0
        dif = 0.0
        n = 0.0
        for indices,_ in training_data.iterrows():
            # indices is a tuple of (userId, movieId)
            n += 1.0
            dif = ratings[indices[0], indices[1]] - self.dot_product(indices[0], indices[1])
            sum += dif**2
        return sum / n
        

    def compute_gradients(self, training_data, ratings):
        num_users = self.num_users
        num_items = self.num_items

        # Initialize gradients with for loops
        grad_user_factors = []
        for i in range(num_users):
            grad_user_factors.append([0.0] * self.n_latency_factors)

        grad_item_factors = []
        for j in range(num_items):
            grad_item_factors.append([0.0] * self.n_latency_factors)

        grad_user_biases = []
        for i in range(num_users):
            grad_user_biases.append(0.0)

        grad_item_biases = []
        for j in range(num_items):
            grad_item_biases.append(0.0)

        # (user_id, item_id) is a tuple of (userId, movieId)
        for (user_id, item_id), _ in training_data.iterrows():
            # Compute prediction (this is based on the CS 472 on linear regression)
            # https://classes.cs.uoregon.edu/24S/cs472/notes/linearRegression.pdf
            # this formula is based on slide 6
            # Also did some more research and found this formula
            
            # May want to add bias before subtracting actual rating
            user_id = int(user_id)
            item_id = int(item_id)

            # Compute y * (w^T x + b)
            z = self.dot_product(user_id, item_id)
            # could be z - ratings[i][j]??
            error = ratings[user_id, item_id] - z
            
            for k in range(self.n_latency_factors):
                grad_user_factors[user_id][k] += -2 * error * self.item_factors[k][item_id] + 2 * self.l2_factor * self.user_factors[user_id][k]
                grad_item_factors[item_id][k] += -2 * error * self.user_factors[user_id][k] + 2 * self.l2_factor * self.item_factors[k][item_id]

            grad_user_biases[user_id] += -2 * error
            grad_item_biases[item_id] += -2 * error

        return grad_user_factors, grad_item_factors, grad_user_biases, grad_item_biases
        
        return grad_w, grad_b

    def l2_regularizer(self, element, l2_factor):
        """Generates gradient L2 regularizer to be added to ________"""
        element
        pass

    def finetune_matrix(self, matrix):
        """Improves the weights of ONE of the factor matrices"""
        pass

    def compute_loss(self, user_id, item_id):
        """Find loss between actual score and predicted score"""
        actual_score = self.user_item_matrix[user_id][item_id]
        pass

    def train(self, training_data, ratings, test_data):
        """Implements ALS, switching between finetuning the user_factors
        and the movie_factors. Stop when both reach MAX_ITER or are converged?"""
        train_losses = []
        test_losses = []
        for iteration in range(self.max_iters):
            # Compute gradients
            grad_user_factors, grad_item_factors, grad_user_biases, grad_item_biases = self.compute_gradients(training_data, ratings)

            # Update user and item factors and biases
            for i in range(len(self.user_factors)):
                for k in range(self.n_latency_factors):
                    self.user_factors[i][k] -= self.eta * grad_user_factors[i][k]
                self.user_biases[i] -= self.eta * grad_user_biases[i]

            for j in range(len(self.item_factors[0])):
                for k in range(self.n_latency_factors):
                    self.item_factors[k][j] -= self.eta * grad_item_factors[j][k]
                self.item_biases[j] -= self.eta * grad_item_biases[j]

            train_loss = self.means_squared_error(training_data, ratings)
            test_loss = self.means_squared_error(test_data, ratings)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print(f"Iteration {iteration + 1}: Train MSE = {train_loss}, Test MSE = {test_loss}")

            # Compute gradient magnitude for convergence check
            gradient_magnitude = sqrt(
                sum(grad_user_factors[i][k] ** 2 for i in range(len(self.user_factors)) for k in range(self.n_latency_factors)) +
                sum(grad_item_factors[j][k] ** 2 for j in range(len(self.item_factors[0])) for k in range(self.n_latency_factors)) +
                sum(grad_user_biases[i] ** 2 for i in range(len(grad_user_biases))) +
                sum(grad_item_biases[j] ** 2 for j in range(len(grad_item_biases)))
            )

            if gradient_magnitude < self.convergence_threshold:
                print(f"Converged at iteration {iteration}")
                break

        return self.user_factors, self.item_factors, self.user_biases, self.item_biases, train_losses, test_losses
    
    def predict_als(self, user_id, item_id):
        """Predict the probability of the correct rating label given the attributes, user_factor[i] & item_factor[i]"""
        return self.dot_product(user_id, item_id)

if __name__=="__main__":
    #This is for testing
    als = ALS_Model(3, 4, 5, 0.001, 50, 0.0001, 0.1, 0.1, 1.0)
    print(als.user_factors)
    print("user biases: ", als.user_biases)
    print(als.item_factors)
    print("item biases:", als.item_biases)
    print(als.n_latency_factors)
    print(als.eta)
    print(als.max_iters)
    print(als.convergence_threshold)
    print(als.l2_factor)

    #Print predicted value
    print(als.dot_product(3, 4))
    print(als.sigmoid_range(als.dot_product(3,4)))

    # Testing with fake simulated data
    all_data = [(i, j) for i in range(als.num_users) for j in range(als.num_items)]
    # Set possible ratings
    possible_ratings = [i / 2.0 for i in range(2, 11)]  # [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    # Generate ratings for approximately 40% of the user-item pairs out of the possible ratings we just created
    # We do this because we want to keep our data sparse, similar to real data, users will not have opinions about every item.
    ratings = {(i, j): random.choice(possible_ratings) for i, j in all_data if random.random() > 0.6}
    
    # Convert to DataFrame
    all_data_df = pd.DataFrame(index=ratings.keys())

    # Split the data into training and test sets
    # You can also add a random_state = parameter to this function to seed splits
    training_data, test_data = train_test_split(all_data_df, test_size=0.3)

    # Train data
    user_factors, item_factors, user_biases, item_biases, train_losses, test_losses = als.train(training_data, ratings, test_data)

    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title('Training and Testing Loss over Iterations')
    plt.show()