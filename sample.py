import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler
    from pyspark.ml.classification import NaiveBayes
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    # 1. Initialize Spark Session
    spark = SparkSession.builder.appName("NaiveBayesPipeline").getOrCreate()

    # 2. Load Sample Data (Example structure)
    # Ensure features are non-negative for Multinomial Naive Bayes
    data = spark.createDataFrame([
        (0, "category_a", 1.0, 2.0),
        (1, "category_b", 2.0, 1.0),
        (0, "category_a", 1.5, 2.5),
        (1, "category_b", 1.0, 0.5)
    ], ["label", "category", "feature1", "feature2"])

    # 3. Define Pipeline Stages
    # Convert categorical strings to numeric indices
    indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")

    # Combine all feature columns into a single vector column
    assembler = VectorAssembler(
        inputCols=["categoryIndex", "feature1", "feature2"], 
        outputCol="features"
    )

    # Initialize the Naive Bayes Estimator
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

    # 4. Construct and Run the Pipeline
    pipeline = Pipeline(stages=[indexer, assembler, nb])
    train_data, test_data = data.randomSplit([0.7, 0.3])
    model = pipeline.fit(train_data)

    # 5. Make Predictions and Evaluate
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    print(f"Model Accuracy: {accuracy}")

    return


if __name__ == "__main__":
    app.run()
