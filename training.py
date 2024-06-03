import pandas as pd
import random
from math import log, exp, sqrt


class ALS_Model():
    def __init__(self, num_users, num_items, n_latency_factors, eta, max_iters, convergence_threshold, l2_factor):
        #self.user_item_matrix = user_item_matrix
        #self.user_matrix = user_matrix
        #self.item_matrix = item_matrix
        self.user_factors = self.randomize_narray(num_users, n_latency_factors)                  #Randomly Initialize
        self.user_biases = self.randomize_narray(2, num_users)                   #Randomly Initialize
        self.item_factors = self.randomize_narray(n_latency_factors, num_items)                 #Randomly Initialize
        self.item_biases = self.randomize_narray(2, num_items)                  #Randomly initialize
        self.n_latency_factors = n_latency_factors
        self.eta = eta
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.l2_factor = l2_factor

    def randomize_narray(self, M:int, N:int):
        empty = []
        if M<=1:
            for el in range(N):
                empty.append(random.gauss(0.0, 1.0))
        else:
            for row in range(M):
                new = []
                for el in range(N):
                    new.append(random.gauss(0.0, 1.0))
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
        num_users = len(self.user_factors)
        print(num_users)
        num_items = len(self.item_factors[0])
        print(num_items)

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
            print("user_id: ", user_id)
            user_id = int(user_id)
            print("item_id: ", item_id)
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

    def train(self, training_data, ratings):
        """Implements ALS, switching between finetuning the user_factors
        and the movie_factors. Stop when both reach MAX_ITER or are converged?"""
        for iteration in range(self.max_iters):
            # Compute gradients
            grad_user_factors, grad_item_factors, grad_user_biases, grad_item_biases = self.compute_gradients(training_data, ratings)

            # Update user and item factors and biases
            for i in range(len(self.user_factors)):
                for k in range(self.n_latency_factors):
                    self.user_factors[i][k] -= self.eta * grad_user_factors[i][k]
                self.user_biases[0][i] -= self.eta * grad_user_biases[i]

            for j in range(len(self.item_factors[0])):
                for k in range(self.n_latency_factors):
                    self.item_factors[k][j] -= self.eta * grad_item_factors[j][k]
                self.item_biases[0][j] -= self.eta * grad_item_biases[j]

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

        return self.user_factors, self.item_factors, self.user_biases, self.item_biases
    
    def predict_als(self, user_id, item_id):
        """Predict the probability of the correct rating label given the attributes, user_factor[i] & item_factor[i]"""
        return self.dot_product(user_id, item_id)
    
    def compute_accuracy(self, test_data, ratings):
        """Compute accuracy of predictions against test data"""
        total_error = 0.0
        for (user_id, item_id), _ in test_data.iterrows():
            user_id = int(user_id)
            item_id = int(item_id)
            predicted_rating = self.predict_als(user_id, item_id)
            actual_rating = ratings[(user_id, item_id)]
            total_error += (predicted_rating - actual_rating) ** 2
        mse = total_error / len(test_data)
        print("Mean Squared Error: ", mse)
        return mse

if __name__=="__main__":
    #This is for testing
    als = ALS_Model(3, 4, 5, 0.01, 20, 0.001, 0.1)
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

    # Sample training data and ratings for testing
    training_data = pd.DataFrame(index=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)])
    ratings = {(0, 0): 4, (0, 1): 3, (1, 0): 4, (1, 1): 5, (2, 2): 3}

    # Train the model
    als.train(training_data, ratings)

    # Compute accuracy
    test_data = pd.DataFrame(index=[(0, 0), (1, 1), (2, 2)])
    als.compute_accuracy(test_data, ratings)

    # grad_user_factors, grad_item_factors, grad_user_biases, grad_item_biases = als.compute_gradients(training_data, l2_factor, ratings)
    # print(grad_user_factors)
    # print(grad_item_factors)
    # print(grad_user_biases)
    # print(grad_item_biases)