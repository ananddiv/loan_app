import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col,when

# 1. Start a Spark session
spark = SparkSession.builder.appName("loan_predict_pyspark").getOrCreate()

# 2. Setup the file paths for input and output data
base_dir = os.path.abspath('/Users/ananddivakaran/Documents/Anand/MDS/loan_app')
print(f'Base Data Directory:{base_dir}')
input_path = f"file://{os.path.join(base_dir, 'data/raw/loan_data.csv')}"
print(f'Input file: {input_path}')
output_path = os.path.join(base_dir, "data/interim/loan_data_interim.csv")
print(f'Output file:{output_path}')
##input_path = "loan_data.csv"

# 3. Load the dataset into a Spark DataFrame
print("--- Data Loading Start ---")
df = spark.read.csv(input_path, header=True, inferSchema=True)

# 4. Check the schema and summary statistics of the dataset
df.printSchema()
print("--- Data Loading Complete ---")

print("--- Data Exploration Start ---")
# 5. Data Preprocessing
# 5.1 Handle Missing Values in LoanAmount & Loam Amount Term Feature - delete the rows.
print("--- Data Preprocessing Start ---")
df_data  = df.na.drop(subset=["LoanAmount","Loan_Amount_Term"])
# 5.3.Handle Missing Values  - imputation with a default value (before splitting the dataset)
#    Married Feature - fill with "No"
#    Dependents Feature - fill with "0"
#    Self_Employed Feature - fill with "No"
#    Credit_History Feature - fill with 0.0
#    Gender Feature - fill with "unknown"   
df_data = df_data.fillna({"Married": "No", "Dependents": "0", "Self_Employed": "No", "Credit_History": 0.0,"Gender":"unknown"})
#df_data.printSchema()
#df_data.summary().show()

df.data_pandas = df_data.toPandas()
null_fileds = df.data_pandas.isnull().sum().reset_index()
null_fileds.columns = ['Column', 'Null_Count']
print(f"Columns with Null Values: {null_fileds[null_fileds['Null_Count'] > 0].shape[0]}")

print("--- Data Preprocessing Complete ---")

# 6. Discretize the below numerical features into categorical features using StringIndexer
#    ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
print("--- Discretization Start ---")

# 6.1 Outliers were detected during the EDA phase in the ApplicantIncome, CoapplicantIncome, Loan Term and LoanAmount features. 
# The discretization process will help mitigate the impact of these outliers on the model's performance by categorizing 
# the continuous values into defined bins, thus reducing the influence of extreme values and allowing the model to focus on broader patterns in the data rather than being skewed by outliers.
# refer the initial EDA code for the outlier detection and visualization steps in the notebook for more details.

# 6.2 Define the bins and labels for discretization
#  Bin the income features into 5 categories: None, Low, Medium, High, Very High
income_bins = [0, 2500, 4000, 6000, float('inf')]
income_labels = ['None', 'Low', 'Medium', 'High', 'Very High']  
# Bin the loan amount and loan term features into 5 categories: None, Low, Medium, High, Very High for loan amount
loan_amount_bins = [0, 100, 200, 300, float('inf')]
loan_amount_labels = ['None', 'Low', 'Medium', 'High', 'Very High']
# Bin the loan term feature into 5 categories: None, Short Term, Medium Term, Long Term, Very Long Term
loan_term_bins = [0, 100, 200, 300, float('inf')]
loan_term_labels = ['None', 'Short Term', 'Medium Term', 'Long Term', 'Very Long Term']

# 6.3 Create a function to discretize a numerical column based on the defined bins and labels
def discretize_column(df, column_name, bins, labels):
    # Create a new column for the discretized values
    discretized_col = f"{column_name}_Category"
    df = df.withColumn(discretized_col, 
                       when(col(column_name) <= bins[0], labels[0])
                       .when((col(column_name) > bins[0]) & (col(column_name) <= bins[1]), labels[1]) 
                       .when((col(column_name) > bins[1]) & (col(column_name) <= bins[2]), labels[2]) 
                       .when((col(column_name) > bins[2]) & (col(column_name) <= bins[3]), labels[3])
                       .otherwise(labels[4]))
    return df
#6.3 Apply the discretization function to the relevant columns in both training and testing datasets
# Apply discretization to training dataset
df_data = discretize_column(df_data, 'ApplicantIncome', income_bins, income_labels)
df_data = discretize_column(df_data, 'CoapplicantIncome', income_bins, income_labels)
df_data = discretize_column(df_data, 'LoanAmount', loan_amount_bins, loan_amount_labels)
df_data = discretize_column(df_data, 'Loan_Amount_Term', loan_term_bins, loan_term_labels)    
print("--- Discretization Complete ---")

df_data.show(1)

# 7. Selecting Features and Target Variable
print("--- Feature Selection Start ---")
# 7.1 Define the feature columns and target column
# Exclude 'Loan_ID' and 'Loan_Status' from the feature columns (Loan ID as it is an identifier and Loan Status as it is the target variable)
categorical_cols = [field.name for field in df_data.schema.fields if field.dataType.typeName() == 'string' and (field.name != 'Loan_Status' and field.name != 'Loan_ID')]
numerical_cols = [field.name for field in df_data.schema.fields if field.dataType.typeName() in ['integer', 'double'] and field.name != 'Credit_History']        
target_cols = 'Loan_Status'

# 7.2. Add credit history as a categorical column since it is binary (1.0/0.0) and acts as a category in this context
categorical_cols.append("Credit_History")

print(f'Categorical Columns: {categorical_cols}')
print(f'Numerical Columns: {numerical_cols}')
print(f'Target Column: {target_cols}')

# Assuming your dataframe is named 'df' and you want to count 'ApplicantIncome'
value_counts_df = df_data.groupBy("ApplicantIncome").count()

# To mimic pandas and sort the results from highest to lowest:
value_counts_df = value_counts_df.orderBy("count", ascending=False)

value_counts_df.show()


print("--- Feature Selection Complete ---")

# 8. Splitting the dataset into training and testing sets
print("--- Data Splitting Start ---")
train_df, test_df = df_data.randomSplit([0.8, 0.2], seed=42)
print("--- Data Splitting Complete ---")

base_stages = []  # List to hold the stages of the pipeline

#9. Indexing and Encoding Categorical Variables
print("--- Indexing and Encoding Start ---")
# 9.1 String Indexing & Encoding for categorical features
for cat_col in categorical_cols:
    print(f"Processing categorical column: {cat_col}")
    indexer = StringIndexer(inputCol=cat_col, outputCol=f"{cat_col}_Index")
    encoder = OneHotEncoder(inputCols=[f"{cat_col}_Index"], outputCols=[f"{cat_col}_Encoded"])     
    base_stages +=  [indexer, encoder]
 
# 9.2 String indexing for the target variable
label_indexer = StringIndexer(inputCol=target_cols, outputCol=f"{target_cols}_Index", handleInvalid='keep')
base_stages += [label_indexer]
print("--- Indexing and Encoding Complete ---")

# 10. Assembling Features into a Single Vector
print("--- Feature Assembling Start ---")
assembler_inputs = [f"{c}_Encoded" for c in categorical_cols]
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Add the assembler to the pipeline stages
base_stages += [assembler]
#print(f"Pipeline Stages: {[stage.__class__.__name__ for stage in base_stages]}")
print("--- Feature Assembling Complete ---")

# 11. Model Building and Evaluation
print("--- Model Building and Evaluation Start ---")
# 11.1 Define the classifiers to be evaluated
classifiers = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol=f"{target_cols}_Index"),
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol=f"{target_cols}_Index", seed = 42),
    "Naive Bayes": NaiveBayes(featuresCol="features", labelCol=f"{target_cols}_Index"),
    "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol=f"{target_cols}_Index", seed = 42)
}
# 12. Training the Models and Comparing 
# 12.1 Create a pipeline for each classifier and evaluate their performance
# Evaluate using Accuracy and F1 Score
evaluator_acc = MulticlassClassificationEvaluator(labelCol=f"{target_cols}_Index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol=f"{target_cols}_Index", predictionCol="prediction", metricName="f1")

output_path = os.path.join(base_dir, "output.txt")
results = {}
# Loop through each classifier, train the model, make predictions, and evaluate the results
for name, classifier in classifiers.items():
    print(f"Training and evaluating: {name}")
    # Create a pipeline with the base stages and the current classifier
    pipeline = Pipeline(stages=base_stages + [classifier])
    # Fit the model on the training data and make predictions on the test data
    model = pipeline.fit(train_df)
    # Make predictions on the test data
    predictions = model.transform(test_df)
    # Evaluate the model's performance using accuracy and F1 score
    accuracy = evaluator_acc.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)
    # Store the results in a dictionary
    results[name] = (accuracy, f1_score)
    print(f"{name} Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}")


# Indentifying the best model based on highest F1 Score
best_accuracy, best_f1_score = results['Random Forest']
print(f"Best Model: Random Forest with Accuracy: {best_accuracy:.4f} and F1 Score: {best_f1_score:.4f}")
    
print("--- Model Building and Evaluation Complete ---")
# Write the results to a file
print(f"Writing results to: {output_path}")
with open(output_path, "w") as f:
    for name, (accuracy, f1_score) in results.items():
        f.write(f"{name} Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}\n") 


