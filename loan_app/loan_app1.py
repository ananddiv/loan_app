import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# 1. Start a Spark session
spark = SparkSession.builder.appName("Loan_Application_Classifier").getOrCreate()


# 2. Load Dataset
base_dir = os.path.abspath('/Users/ananddivakaran/Documents/Anand/MDS/loan_app')
print(f'Base Data Directory:{base_dir}')
input_path = f"file://{os.path.join(base_dir, 'data/raw/loan_data.csv')}"
print(f'Input file: {input_path}')
output_path = os.path.join(base_dir, "/data/interim/loan_data_interim.csv")
print(f'Output file:{output_path}')
##input_path = "loan_data.csv"
df = spark.read.csv(input_path, header=True, inferSchema=True)

# 3. Data Cleaning: Drop Loan_ID (not a feature) and drop missing values
df = df.drop("Loan_ID").dropna()

# 4. Feature Separation
# Identify categorical and numerical columns automatically
categorical_cols = [field.name for field in df.schema.fields if field.dataType.typeName() == 'string' and field.name != 'Loan_Status']
numerical_cols = [field.name for field in df.schema.fields if field.dataType.typeName() in ['integer', 'double'] and field.name != 'Credit_History']

print(f'Categorical Columns: {categorical_cols}')
print(f'Numerical Columns: {numerical_cols}')

# Note: Credit_History is technically numeric (1.0/0.0) but acts as a category. We'll leave it as numeric for this implementation.
numerical_cols.append("Credit_History") 

# 5. Build the Preprocessing Stages
stages = []

# Index and One-Hot Encode Categorical Columns
for cat_col in categorical_cols:
    indexer = StringIndexer(inputCol=cat_col, outputCol=f"{cat_col}_Index")
    encoder = OneHotEncoder(inputCols=[f"{cat_col}_Index"], outputCols=[f"{cat_col}_OHE"])
    stages += [indexer, encoder]

print(f'stages: {stages}')

# Index the Target Variable
label_indexer = StringIndexer(inputCol="Loan_Status", outputCol="label")
stages += [label_indexer]

# Assemble all features into a single vector
assembler_inputs = [f"{c}_OHE" for c in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
stages += [assembler]

# 6. Define the Logistic Regression Model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
stages1 = stages + [lr]

# 7. Create Pipeline and Split Data
pipeline = Pipeline(stages=stages1)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 8. Train Model
print("Training Logistic Regression Model...")
lr_model = pipeline.fit(train_data)

# 9. Evaluate Model
predictions = lr_model.transform(test_data)

# Calculate Accuracy
acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = acc_evaluator.evaluate(predictions)

# Calculate F1 Score (PySpark defaults to weighted F1)
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Logistic Regression F1 Score: {f1_score:.4f}")

rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42, numTrees=100)
stages2 = stages + [rf]

pipeline = Pipeline(stages=stages2)

# 8. Train Model
print("Training Random Forest Model...")
rf_model = pipeline.fit(train_data)

# 9. Evaluate Model
predictions = rf_model.transform(test_data)

# Calculate Accuracy
acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = acc_evaluator.evaluate(predictions)

# Calculate F1 Score
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)

print(f"Random Forest Accuracy: {accuracy:.4f}")
print(f"Random Forest F1 Score: {f1_score:.4f}")

nb = NaiveBayes(labelCol="label", featuresCol="features", modelType="gaussian")
stages3 = stages + [nb]

# 7. Create Pipeline and Split Data
pipeline = Pipeline(stages=stages3)

# 8. Train Model
print("Training Naive Bayes Model...")
nb_model = pipeline.fit(train_data)

# 9. Evaluate Model
predictions = nb_model.transform(test_data)

# Calculate Accuracy
acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = acc_evaluator.evaluate(predictions)

# Calculate F1 Score
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)

print(f"Naive Bayes Accuracy: {accuracy:.4f}")
print(f"Naive Bayes F1 Score: {f1_score:.4f}")

spark.stop()