import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

class RecommenderSystem:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.U = None
        self.sigma = None
        self.Vt = None
        
    def create_matrix(self, ratings_df):
        """
        Create user-item matrix from ratings dataframe
        """
        if 'user_id' not in ratings_df.columns or 'item_id' not in ratings_df.columns or 'rating' not in ratings_df.columns:
            raise ValueError("Ratings dataframe must contain 'user_id', 'item_id', and 'rating' columns")
        
        self.user_item_matrix = ratings_df.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)
        
    def compute_similarity(self):
        """
        Compute user-user and item-item similarity matrices
        """
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix is not created. Call create_matrix first.")
            
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        
    def matrix_factorization(self, k=20):
        """
        Perform matrix factorization using SVD
        """
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix is not created. Call create_matrix first.")
            
        # Check if k is valid
        if k <= 0 or k > min(self.user_item_matrix.shape):
            raise ValueError(f"k must be positive and not greater than min(user_count, item_count). Got k={k}")
            
        # Normalize the ratings
        ratings_mean = np.mean(self.user_item_matrix.values, axis=1)
        ratings_norm = self.user_item_matrix.values - ratings_mean.reshape(-1, 1)
        
        # Perform SVD
        U, sigma, Vt = svds(ratings_norm, k=k)
        # Reverse to descending order
        U, sigma, Vt = U[:, ::-1], sigma[::-1], Vt[::-1, :]
        
        # Convert to diagonal matrix
        sigma_diag = np.diag(sigma)
        
        # Calculate predicted ratings
        predicted_ratings = np.dot(np.dot(U, sigma_diag), Vt) + ratings_mean.reshape(-1, 1)
        # Return as DataFrame for easier use
        return pd.DataFrame(predicted_ratings, index=self.user_item_matrix.index, columns=self.user_item_matrix.columns)
    
    def get_user_recommendations(self, user_id, n_recommendations=5):
        """
        Get top N recommendations for a user using collaborative filtering
        """
        if self.user_similarity is None:
            raise ValueError("User similarity matrix is not computed. Call compute_similarity first.")
            
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User ID {user_id} not found in the user-item matrix")
            
        if n_recommendations <= 0:
            raise ValueError("Number of recommendations must be positive")
            
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_similarities = self.user_similarity[user_idx]
        
        # Get similar users (use n_recommendations+1 to be flexible)
        similar_users = np.argsort(user_similarities)[::-1][1:n_recommendations+1]
        
        # Get items that user hasn't rated
        user_unrated = self.user_item_matrix.iloc[user_idx] == 0
        
        # Check if user has unrated items
        if not any(user_unrated):
            return []  # User has rated all items
            
        # Calculate predicted ratings
        recommendations = []
        for item_id in self.user_item_matrix.columns[user_unrated]:
            item_ratings = self.user_item_matrix[item_id].iloc[similar_users]
            # Only consider items with ratings
            if np.sum(item_ratings) > 0:
                predicted_rating = np.average(item_ratings, weights=user_similarities[similar_users])
                recommendations.append((item_id, predicted_rating))
            
        # Sort and return top N recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_similar_items(self, item_id, n_similar=5):
        """
        Get N most similar items to a given item
        """
        if self.item_similarity is None:
            raise ValueError("Item similarity matrix is not computed. Call compute_similarity first.")
            
        if item_id not in self.user_item_matrix.columns:
            raise ValueError(f"Item ID {item_id} not found in the user-item matrix")
            
        if n_similar <= 0:
            raise ValueError("Number of similar items must be positive")
            
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        item_similarities = self.item_similarity[item_idx]
        
        # Get similar items
        similar_items = np.argsort(item_similarities)[::-1][1:n_similar+1]
        return [(self.user_item_matrix.columns[idx], item_similarities[idx]) 
                for idx in similar_items]

# Example usage:
if __name__ == "__main__":
    # Sample data
    ratings_data = {
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'item_id': [1, 2, 2, 3, 1, 3, 1, 2],
        'rating': [5, 3, 4, 5, 3, 4, 4, 5]
    }
    ratings_df = pd.DataFrame(ratings_data)
    
    # Initialize recommender system
    recommender = RecommenderSystem()
    
    # Create user-item matrix
    recommender.create_matrix(ratings_df)
    
    # Compute similarities
    recommender.compute_similarity()
    
    # Get recommendations for user 1
    recommendations = recommender.get_user_recommendations(1)
    print("Recommendations for user 1:", recommendations)
    
    # Get similar items to item 1
    similar_items = recommender.get_similar_items(1)
    print("Similar items to item 1:", similar_items)
    
    # Perform matrix factorization
    predicted_ratings = recommender.matrix_factorization(k=2)
    print("Predicted ratings matrix shape:", predicted_ratings.shape)
    print(predicted_ratings)