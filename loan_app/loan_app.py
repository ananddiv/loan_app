# 1. Import libraries
import pyspark 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

# 2. Start a spark session
spark = SparkSession.builder.appName("Loan_Application_Classifier").getOrCreate()

# 3. Setup the base path
base_dir = os.path.abspath('/Users/ananddivakaran/Documents/Anand/MDS/loan_app')
print(f'Base Data Directory:{base_dir}')
input_path = f"file://{os.path.join(base_dir, 'data/raw/loan_data.csv')}"
print(f'Input file: {input_path}')
output_path = os.path.join(base_dir, "/data/interim/loan_data_interim.csv")
print(f'Output file:{output_path}')

# 4. Load Dataset
# Read data
data = spark.read.csv(input_path, header=True, inferSchema=True)

# 5. Drop the columns which are not required for modeling
columns_to_keep= ['Married','Credit_History','Property_Area','Loan_Status']
data = data.select(columns_to_keep)

# 6. Drop rows with missing values
df = data.dropna()
print(df.count())

# 7. Convert the target variable to numeric
label_indexer = StringIndexer(inputCol="Loan_Status", outputCol="Loan_Status_Index")
df = label_indexer.fit(df).transform(df)
df = df.drop("Loan_Status").withColumnRenamed("Loan_Status_Index", "Loan_Status")

# 8. Converting categories to numeric indices
indexer_married = StringIndexer(inputCol="Married", outputCol="indexed_married")
indexer_credit_history = StringIndexer(inputCol="Credit_History", outputCol="indexed_credit_history")
indexer_property_area = StringIndexer(inputCol="Property_Area", outputCol="indexed_property_area")

indexers = [indexer_married, indexer_credit_history, indexer_property_area]
# 9. Encode the categorical features
encoder = OneHotEncoder(inputCols=["indexed_married", "indexed_credit_history","indexed_property_area"],
                        outputCols=["encoded_married", "encoded_credit_history","encoded_property_area"])

# 10. Combine all the encoded features to one vector
assembler = VectorAssembler(
    inputCols=["encoded_married", "encoded_credit_history","encoded_property_area"],
    outputCol="loan_features")

# 11. Split the data into training and testing sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 12. Create a pipeline to apply the transformations
pipeline = Pipeline(stages=[indexer_married, indexer_credit_history, indexer_property_area, encoder, assembler])

# 13. Fit the pipeline to the training data
base_model = pipeline.fit(train_data)
print("✅ Pipeline Fitting Complete!")

# 14. Transform the training and testing data using the fitted pipeline
train_data_transformed = base_model.transform(train_data)
test_data_transformed = base_model.transform(test_data)

# 15. Initialize the classifiers
logistic_regression = LogisticRegression(labelCol="Loan_Status", featuresCol="loan_features")
decision_tree_classifier = DecisionTreeClassifier(labelCol="Loan_Status", featuresCol="loan_features", seed = 42)
random_forest_classifier = RandomForestClassifier(labelCol="Loan_Status", featuresCol="loan_features", seed = 42)


print("\n--- Starting Model Training (Logistic Regression Estimator) ---")
# Fit the pipeline to the entire dataset (Training phase)
log_reg_model = logistic_regression.fit(train_data_transformed)
print("✅ Model Training Complete!")
log_reg_predictions = log_reg_model.transform(test_data_transformed)    
print("Model Evaluation Complete!")
evaluator = MulticlassClassificationEvaluator(labelCol="Loan_Status",
predictionCol="prediction", metricName="accuracy")
LogisticRegression_accuracy = evaluator.evaluate(log_reg_predictions)
print(f"Logistic Regression Accuracy: {LogisticRegression_accuracy:.4f}")

print("\n--- Starting Model Training (Decision Tree Estimator) ---")
# Fit the pipeline to the entire dataset (Training phase)
decision_tree_model = decision_tree_classifier.fit(train_data_transformed)
print("✅ Model Training Complete!")
dt_predictions = decision_tree_model.transform(test_data_transformed)    
print("Model Evaluation Complete!")
evaluator = MulticlassClassificationEvaluator(labelCol="Loan_Status",
predictionCol="prediction", metricName="accuracy")
DecisionTree_accuracy = evaluator.evaluate(dt_predictions)
print(f"Decision Tree Accuracy: {DecisionTree_accuracy:.4f}")


output_path = os.path.join(base_dir, "model_performance.txt")
with open(output_path, "w") as f:
    f.write(f"Logistic Regression Classification Accuracy:{LogisticRegression_accuracy:.4f}\n")
    #f.write(f"Random Forest Classification Accuracy:{RandomForest_accuracy:.4f}\n")
    f.write(f"Decision Tree Classification Accuracy:{DecisionTree_accuracy:.4f}\n")
# Stop the Spark session
spark.stop()