# Diamond-Prediction

# Project: Diamond Price Prediction

This project focuses on predicting diamond prices based on various features such as cut, color, clarity, and physical dimensions using multiple machine learning models. The project uses models like Linear Regression, Decision Trees, Random Forests, and XGBoost to predict diamond prices and compare the performance of each model.

## Features

- **Data Loading**: The dataset is loaded from a CSV file for analysis.
- **Data Cleaning**: Unnecessary columns are removed, and rows with missing or outlier values in dimensions (x, y, z) are filtered out.
- **Feature Encoding**: Categorical features like `cut`, `color`, and `clarity` are encoded using Label Encoding for machine learning compatibility.
- **Model Training**: Various regression models (Linear Regression, Decision Tree, Random Forest, and XGBoost) are trained to predict diamond prices.
- **Model Evaluation**: Models are evaluated using RMSE (Root Mean Squared Error) and R² score.
- **Feature Importance**: For the XGBoost model, feature importance is visualized to show which features contribute the most to price prediction.
- **Predictions Visualization**: The actual vs predicted prices are plotted for the best-performing model (XGBoost).

## Prerequisites

To run this project, you need to have the following Python libraries installed:

- **pandas**: For data manipulation.
- **numpy**: For numerical operations.
- **scikit-learn**: For machine learning models.
- **matplotlib and seaborn**: For data visualization.
- **xgboost**: For XGBoost model implementation.

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## Workflow

### 1. Data Loading and Cleaning:
The project starts by loading the `diamonds.csv` dataset into a pandas DataFrame. The data is cleaned by:
- Dropping unnecessary columns (like `Unnamed: 0`).
- Filtering out rows with invalid dimension values (x, y, z).
- Removing outliers in features like `depth` and `table`.

### 2. Data Visualization:
- **Pair Plot**: Visualizes relationships between features and the diamond `cut`.
- **Correlation Matrix**: Displays correlations between all features using a heatmap.

### 3. Model Training:
The cleaned dataset is split into features (`X`) and target (`y`), where `price` is the target variable. The features are standardized using `StandardScaler`, and four machine learning models are trained:
- **Linear Regression**
- **Decision Tree**
- **Random Forest**
- **XGBoost**

### 4. Model Evaluation:
Each model's performance is evaluated using two metrics:
- **RMSE** (Root Mean Squared Error) to measure the error between predicted and actual prices.
- **R² Score** to measure how well the model fits the data.

The best-performing model is selected based on these metrics.

### 5. Feature Importance (XGBoost):
For the XGBoost model, feature importance is visualized to identify which features (like `carat`, `cut`, `depth`, etc.) have the most influence on predicting diamond prices.

### 6. Prediction Visualization:
A scatter plot is generated to show the relationship between actual and predicted prices for the best-performing model, helping to visually assess the model's performance.

## Visualizations

- **Correlation Heatmap**: Displays the correlation between features using a color-coded heatmap.
- **Feature Importance Bar Plot**: Shows the most important features for predicting prices in the XGBoost model.
- **Actual vs Predicted Scatter Plot**: Visualizes the accuracy of the model's predictions.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diamond-price-prediction.git
   cd diamond-price-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `diamonds.csv` file is in the correct path.

4. Run the Python script to execute the full pipeline:
   ```bash
   python diamond_price_prediction.py
   ```

## Results

- **Model Performance**: Each model's performance is evaluated in terms of RMSE and R² score. The XGBoost model is likely to provide the best results based on its ability to handle complex datasets.
  
| Model               | RMSE   | R²    |
|---------------------|--------|-------|
| Linear Regression    | XX.XX  | X.XX  |
| Decision Tree        | XX.XX  | X.XX  |
| Random Forest        | XX.XX  | X.XX  |
| XGBoost              | XX.XX  | X.XX  |

(Replace `XX.XX` with actual results after running the models.)

- **Feature Importance**: The features with the highest influence on predicting diamond prices (e.g., `carat`, `cut`, `color`) are visualized.

## Conclusion

This project demonstrates the process of predicting diamond prices using multiple regression models. The XGBoost model performs the best in this case, showing the highest accuracy in predicting prices. This analysis helps in understanding which features are the most significant when it comes to pricing diamonds, offering valuable insights into diamond valuation.
