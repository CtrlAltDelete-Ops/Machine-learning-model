This Movie Recommendation System suggests movies based on user ratings. It uses Python with libraries like Pandas, Scikit-learn, and Streamlit. The system works by loading data on movie ratings and titles, calculating movie similarity using cosine similarity, and recommending movies similar to the one you input.

The steps are as follows:

Data Loading: The program loads two datasets—one for user ratings and one for movie titles.
Data Merging: It merges the datasets so each rating is linked to the correct movie title.
User-Movie Matrix: It creates a matrix showing how each user rated each movie.
Cosine Similarity: The system calculates the similarity between movies based on user ratings.
Recommendations: Based on the similarity, the system suggests the top 5 movies similar to the one you enter.
Evaluation: The system evaluates its performance using Precision, Recall, and F1 Score, comparing its predictions with actual ratings.(the evaluation is commented out for now as I have had a few erorrs related to this)
You can use the Streamlit web interface to input a movie title and get recommendations. To run the program, you need the dataset and the required libraries installed. The system also evaluates how accurate its recommendations are based on a test set of ratings.
