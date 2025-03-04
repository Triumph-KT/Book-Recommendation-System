# Book Recommendation System Using Rank-Based, Collaborative Filtering, and Matrix Factorization

Developed and optimized a hybrid book recommendation system using Rank-Based, Similarity-Based Collaborative Filtering, and Matrix Factorization (SVD) models to deliver personalized book recommendations, minimize irrelevant suggestions, and enhance user engagement on large-scale, sparse datasets.

## Project Overview

This project addresses the challenge of recommending books to users based on historical ratings and interactions. By leveraging multiple recommendation strategies, the system delivers tailored suggestions while overcoming common challenges such as the cold-start problem and data sparsity, improving customer satisfaction and retention for online bookstores or digital libraries.

## Dataset

The **Book-Crossing dataset** includes:
- **Ratings Dataset**:
  - `user_id`: Unique identifier for users.
  - `ISBN`: International Standard Book Number.
  - `book_rating`: Rating on a scale from 0 to 10.
- **Books Dataset**:
  - `ISBN`: Identifier for books.
  - `Book-title`: Title of the book.
  - `Book-author`: Author's name.
  - `Year-of-Publication`: Year published.
  - `Publisher`: Publisher name.
  - `Image URLs`: Links to cover images.
- **Users Dataset**: Contains user demographics.

Key characteristics:
- **185,973 books** and **77,805 users**.
- **433,671 total ratings**, resulting in a highly **sparse matrix**.

## Objectives

- Build a hybrid recommendation system:
  - Rank-Based Recommendations for new users.
  - Collaborative Filtering using User-User and Item-Item KNN.
  - Matrix Factorization with SVD.
- Maximize relevant recommendations through precision and recall tuning.
- Minimize false negatives (missed relevant books) and false positives (irrelevant books).
- Optimize model performance via hyperparameter tuning.
- Provide business insights for production deployment.

## Methods

### Data Preprocessing:
- Cleaned and merged ratings, books, and users datasets.
- Applied **Label Encoding** for categorical features.
- Normalized ratings and handled sparsity by filtering inactive users/books.
- Corrected ratings by adjusting for the number of interactions per book.

### Model Development:

#### 1. **Rank-Based Recommendation System**:
- Recommends top-rated books across all users.
- Useful for new users with no prior history (cold-start).

#### 2. **Similarity-Based Collaborative Filtering**:
- **User-User KNN**:
  - Baseline RMSE: **1.84**, F1-Score@10: **0.81**.
  - Optimized RMSE: **1.68**, F1-Score@10: **0.86**.
  - Outperformed all other models.
- **Item-Item KNN**:
  - Baseline RMSE: **1.62**, F1-Score@10: **0.80**.
  - Optimized RMSE: **1.58**, slight improvement in F1-Score.

#### 3. **Matrix Factorization (SVD)**:
- Applied SVD to capture latent user and book features.
- Baseline F1-Score better than item-based models but lower than optimized user-user KNN.
- Limited improvement through hyperparameter tuning.

### Evaluation:
- Measured performance using:
  - **RMSE (Root Mean Squared Error)**.
  - **Precision@10**, **Recall@10**, and **F1-Score@10**.
- Tracked **False Negatives** and **False Positives** to balance business impacts of missed or irrelevant recommendations.
- Hyperparameter tuning via **GridSearchCV** for all models.

## Results

| Model                   | RMSE  | Precision@10 | Recall@10 | F1-Score@10 |
|-------------------------|-------|--------------|-----------|-------------|
| User-User KNN (baseline)| 1.84  | 0.81         | 0.81      | 0.81        |
| User-User KNN (tuned)   | 1.68  | **0.86**     | **0.86**  | **0.86**    |
| Item-Item KNN (baseline)| 1.62  | 0.80         | 0.80      | 0.80        |
| Item-Item KNN (tuned)   | 1.58  | Slight gain  | Slight gain | ~0.81     |
| SVD (baseline)          | —     | Moderate     | Moderate  | Lower than optimized KNN |

- **User-User KNN (optimized)** achieved the best balance of relevance and accuracy with **F1-Score@10 = 0.86**.
- Rank-based models provided strong cold-start solutions but lacked personalization.
- SVD models provided modest improvements but were outperformed by optimized KNN.

## Business/Scientific Impact

- Increased customer engagement through precise and relevant recommendations.
- Recommended:
  - Deploying **Rank-Based Recommendations** for new users.
  - Using **Optimized User-User KNN** for returning users with interaction history.
  - Regular model retraining to account for new users, books, and rating data.
  - Campaigns to encourage user ratings to mitigate data sparsity.
- Minimized missed recommendations (False Negatives), boosting customer retention.
- Enhanced recommendation quality, supporting increased sales and reduced browsing time.

## Technologies Used

- Python
- Scikit-learn
- Surprise Library (SVD, KNN)
- Pandas
- NumPy
- Matplotlib
- Seaborn
- GridSearchCV

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/book-recommendation-system.git
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open and run the notebook to:
   - Preprocess datasets.
   - Train Rank-Based, KNN, and SVD models.
   - Evaluate model performance.
   - Analyze top-K recommendations and misclassification patterns.

## Future Work

- Introduce **neural network-based recommenders** (e.g., **Neural Collaborative Filtering**).
- Apply **SMOTE** or synthetic data generation to handle sparsity.
- Develop an interactive web application with real-time recommendations.
- Integrate user feedback loops to adapt recommendations dynamically.
- Expand feature engineering to include book genres, tags, and user reviews.
