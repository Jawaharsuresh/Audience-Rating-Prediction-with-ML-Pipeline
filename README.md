This project predicts the audience rating of movies based on available data using a machine learning pipeline. We utilize various tools and libraries to clean, preprocess, 
and build a predictive model that estimates the audience rating for movies. The project breaks down into several steps including data loading, cleaning, encoding, building the model, 
and evaluating its performance.

Key Features:
Data Preprocessing: Handle missing values, encode categorical data, and scale features for better model performance.
Random Forest Model: The predictive model used to estimate the audience rating is a Random Forest Regressor.
Visualization: Visualize predictions using scatter plots comparing actual vs. predicted ratings.
Model Evaluation: Evaluate the model's performance using metrics like Mean Absolute Error (MAE) and R² score.
Prerequisites:
Make sure you have the following libraries installed to run the project:
pandas: Data manipulation and analysis
scikit-learn: Machine learning library for building models and evaluation
matplotlib: For creating visualizations
seaborn: For enhanced graphical representation
You can install the necessary packages using the following command: bash

pip install pandas scikit-learn matplotlib seaborn
How to Run:
Upload Dataset: First, upload the dataset (Rotten_Tomatoes_Movies3.xls) to Google Colab via the left-side bar (File > Upload File).
Run the Code: Once the dataset is uploaded, you can run the following steps to load, clean, train the model, and evaluate it.
Step-by-Step Breakdown:
Import Libraries: Import all necessary Python libraries for data processing, machine learning, and visualization.

Load Data: Load the dataset using pandas.

python
data = load_data("path_to_your_file")
Clean Data: Handle missing values by removing rows with NaN values.

data = handle_missing_values(data)
Encode Categorical Data: Convert text columns to numerical values using LabelEncoder.


data = encode_categorical_columns(data)
Split Data: Split the data into features (X) and target variable (y) and then perform an 80/20 split for training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Build the Model: Use a RandomForestRegressor model with a pipeline that scales the data before training the model.
pipeline = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))])
pipeline.fit(X_train, y_train)
Visualize Predictions: Create a scatter plot comparing actual vs predicted ratings.
visualize_predictions(y_test, y_pred)
Model Evaluation: Print evaluation metrics like Mean Absolute Error and R² Score.
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"R² Score: {r2_score(y_test, y_pred)}")
Dataset:
The dataset (Rotten_Tomatoes_Movies3.xls) contains various features related to movies, such as titles, genre, director, and audience ratings. 
The goal of the project is to predict the audience_rating based on these features.

Example Output:
After running the project, the output includes:

The first few rows of the dataset showing both actual and predicted audience ratings.
A visualization comparing the actual ratings vs predicted ratings in a scatter plot.
Model validation metrics such as Mean Absolute Error and R² Score to evaluate model performance.
Conclusion:
This project demonstrates how to build an end-to-end machine learning pipeline to predict movie audience ratings. It shows how to handle real-world data, preprocess it, build a model, 
and visualize results. While the model provides valuable insights, there is always room for improvement, such as fine-tuning the model or adding more features to enhance prediction accuracy.

Future Work:
Explore other machine learning models to improve prediction accuracy.
Implement hyperparameter tuning to optimize the model’s performance.

Add more features to the dataset (such as movie budget, runtime, etc.) to improve the model's predictions.
