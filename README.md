
# Ride Hailing Price Prediction

This project focuses on predicting ride-hailing prices using machine learning models. The goal is to analyze and forecast prices based on various factors such as distance, time of day, traffic conditions, and other relevant features.
## Authors

This project was developed by [Dendi Apriyandi](https://www.linkedin.com/in/dendiapriyandi), an entry level data analyst and business intelligence.
## Dataset

[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma)
## Table of Contents

- [Introduction](#introduction)
- [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Modeling Approach](#modeling-approach)
- [Tools and Technologies](#tools-and-technologies)
- [Results](#results)
- [Insight](#insight)
- [Further Analysis](#further-analysis)

## Introduction

The project aims to predict the price of ride-hailing services based on multiple features using machine learning models. Various regression techniques such as Linear Regression and ElasticNet Regression are explored and compared based on performance metrics like MAE, MSE, RMSE, and R2 score.
## Data Description

The dataset contains the following key features:

- Features: Distance, time of day, traffic conditions, and more.

- Target Variable: Ride price (in local currency).
## Data Preprocessing
Data preprocessing involved several steps:
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Train-test split
## Modeling Approach

Modeling Approach
The following models were implemented and evaluated:

- Linear Regression: A simple baseline model to predict the price based on a linear relationship between features and target.

- ElasticNet Regression: A combination of Lasso and Ridge regression to improve predictions by handling multicollinearity and performing feature selection.
## Tools and Technologies

This project uses the following tools and technologies:

- **Python 3.x**: The core programming language used for data manipulation and machine learning.
- **Jupyter Notebook**: An interactive environment for running and visualizing code.
- **Pandas**: Used for data manipulation and analysis.
- **NumPy**: Used for numerical computations.
- **Matplotlib**: Used for data visualization.
- **Scikit-learn**: The main machine learning library used for building and evaluating models.
- **ElasticNet**: A regression model that combines Lasso and Ridge regularization.
- **Linear Regression**: A basic regression model to establish a baseline.

These tools were essential for data processing, model building, evaluation, and visualization throughout the project.
## Results

- Linear Regression:

    - MAE (Train): 1.84
    - MAE (Test): 1.83
    - R² (Test): 0.91

- ElasticNet Regression (with alpha=0.01 and l1_ratio=0.8):

    - MAE (Train): 1.92
    - MAE (Test): 1.91
    - R² (Test): 0.89
    
The ElasticNet model demonstrates slightly better performance in terms of regularization, but the Linear Regression model still holds competitive accuracy.
## Insight

Throughout the development of this project, several key insights were gathered:

1. **Feature Importance**: By using models like ElasticNet, it becomes clear that not all features contribute equally to the prediction. ElasticNet’s ability to perform feature selection highlights the importance of regularization in models with many features or potential multicollinearity.
   
2. **Model Selection**: While Linear Regression provides a solid baseline with a high R² score, more advanced models like ElasticNet help prevent overfitting and improve the robustness of the model by reducing the impact of less significant features.

3. **Regularization Impact**: ElasticNet's regularization (combining both Lasso and Ridge) demonstrates how adding penalties to large coefficients helps simplify models and make them more generalizable to unseen data, especially when dealing with multicollinear features.

4. **Model Evaluation Metrics**: Metrics such as MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) show how well models generalize to new data. A model with lower RMSE and MAE scores is generally preferred, though regularization models might show slightly higher errors but offer better interpretability and stability in the long run.

5. **Practical Applications**: This kind of model could be highly beneficial for ride-hailing companies to optimize their pricing strategies based on various factors like time of day, traffic, and distance, ensuring a fair and competitive pricing model that balances demand and revenue.

These insights reflect the importance of selecting the right model based on the problem at hand and the trade-offs between model complexity, interpretability, and performance.
## Further Analysis

Further exploration and improvements to the model could include:

1. **Feature Engineering**: Exploring various feature engineering techniques such as polynomial features, interaction terms, and binning of continuous variables to capture more complex relationships between features and the target variable.

2. **Hyperparameter Tuning**: Performing a comprehensive hyperparameter tuning, particularly for the ElasticNet model, to optimize performance. This could involve adjusting parameters like `alpha` and `l1_ratio` to find the ideal balance between Lasso and Ridge regularization.

3. **Handling Outliers and Data Imbalance**: Investigating the impact of outliers and addressing potential data imbalances (such as skewed ride length distributions) could help in improving model robustness. Techniques like trimming or transforming outliers, or using sampling methods to address imbalance, could be applied.

4. **Generalization on Unseen Data**: Evaluating model performance on unseen data (via a dedicated test set or cross-validation) to ensure that the model generalizes well and is not overfitting to the training data.

5. **Alternative Regression Models**: Considering other regression models, such as **Random Forest** or **Gradient Boosting**, which may offer better performance for non-linear data or in the presence of many complex interactions between features.

By conducting this further analysis, the model can be made more robust and accurate, potentially leading to significant improvements in predictive performance and applicability to real-world scenarios.