import pandas as pd
import random
from math import log, exp


class ALS_Model():
    def __init__(self, num_users, num_items, n_latency_factors):
        #self.user_item_matrix = user_item_matrix
        #self.user_matrix = user_matrix
        #self.item_matrix = item_matrix
        self.user_factors = self.randomize_narray(num_users, n_latency_factors)                  #Randomly Initialize
        self.user_biases = self.randomize_narray(1, num_users)                   #Randomly Initialize
        self.item_factors = self.randomize_narray(n_latency_factors, num_items)                 #Randomly Initialize
        self.item_biases = self.randomize_narray(1, num_items)                  #Randomly initialize
        self.n_latency_factors = n_latency_factors

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
            #indices is a tuple of (userId, movieId)
            n += 1.0
            dif = ratings[indices[0], indices[1]] - self.dot_product(indices[0], indices[1])
            sum += dif**2
        return sum / n
        

            

    def l2_regularizer(self, element, l2_factor):
        """Generates gradient L2 regularizer to be added to ________"""
        pass

    def finetune_matrix(self, matrix):
        """Improves the weights of ONE of the factor matrices"""
        pass

    def compute_loss(self, user_id, item_id):
        """Find loss between actual score and predicted score"""
        actual_score = self.user_item_matrix[user_id][item_id]
        pass

    def train(self):
        """Implements ALS, switching between finetuning the user_factors
        and the movie_factors. Stop when both reach MAX_ITER or are converged?"""

if __name__=="__main__":
    #This is for testing
    als = ALS_Model(3, 4, 5)
    print(als.user_factors)
    print(als.user_biases)
    print(als.item_factors)
    print(als.item_biases)
    #Print predicted value
    print(als.dot_product(3, 4))
    print(als.sigmoid_range(als.dot_product(3,4)))