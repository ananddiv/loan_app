import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model-2 - Logistic Regression
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Import python libraries
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pyspark 
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import StringIndexer, VectorAssembler
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

    return (
        MulticlassClassificationEvaluator,
        OneHotEncoder,
        Pipeline,
        RandomForestClassifier,
        SparkSession,
        StringIndexer,
        VectorAssembler,
        mo,
        os,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Start a spark session
    """)
    return


@app.cell
def _(SparkSession):
    spark = SparkSession.builder.appName("Loan_Application_Classifier").getOrCreate()
    return (spark,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Setup file path
    """)
    return


@app.cell
def _(os):
    # Setup the base path
    base_dir = os.path.abspath('/Users/ananddivakaran/Documents/Anand/MDS/loan_app')
    output_path_train = os.path.join(base_dir, "data/processed/loan_data_train")
    print(f'Train file:{output_path_train}')
    output_path_test = os.path.join(base_dir, "data/processed/loan_data_test")
    print(f'Test file:{output_path_test}')
    return output_path_test, output_path_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Import Train and Test Files
    """)
    return


@app.cell
def _(output_path_test, output_path_train, spark):
    spark_train = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(output_path_train)

    spark_test = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(output_path_test)
    return spark_test, spark_train


@app.cell
def _(spark_test, spark_train):
    df_train = spark_train.toPandas()
    df_test = spark_test.toPandas()
    return (df_train,)


@app.cell
def _(df_train):
    df_train.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Setup the Predator and Target Variables
    """)
    return


@app.cell
def _(
    OneHotEncoder,
    Pipeline,
    RandomForestClassifier,
    StringIndexer,
    VectorAssembler,
    spark_train,
):
    Label_col = "Loan_Status"
    Predator_col = ["Married", "Credit_History","Property_Area"]

    # Converting categories to numeric indices
    indexer_married = StringIndexer(inputCol="Married", outputCol="indexed_married")
    indexer_credit_history = StringIndexer(inputCol="Credit_History", outputCol="indexed_credit_history")
    indexer_property_area = StringIndexer(inputCol="Property_Area", outputCol="indexed_property_area")

    # Converting numeric indices to sparce binary vectors
    encoder_married = OneHotEncoder(inputCol="indexed_married",outputCol="encoded_married")
    encoder_credit_history = OneHotEncoder(inputCol="indexed_credit_history",outputCol="encoded_credit_history")
    encoder_property_area = OneHotEncoder(inputCol="indexed_property_area",outputCol="encoded_property_area")

    # Combine all the features to one vector
    assembler = VectorAssembler(
        inputCols=["encoded_married", "encoded_credit_history","encoded_property_area"],
        outputCol="features"
    )

    # The Pipeline orchestrates all steps: Indexing -> Encoding -> Assembling -> Training
    pipeline = Pipeline(stages=[
        indexer_married,
        encoder_married,
        indexer_credit_history,
        encoder_credit_history,
        indexer_property_area,
        encoder_property_area,
        assembler,
        RandomForestClassifier(labelCol=Label_col, featuresCol="features", seed=42) # The final model stage
    ])

    print("\n--- Starting Model Training (Fitting the Pipeline) ---")
    # Fit the pipeline to the entire dataset (Training phase)
    pipelineModel = pipeline.fit(spark_train)
    print("✅ Model Training Complete!")
    return Label_col, pipelineModel


@app.cell
def _(Label_col, MulticlassClassificationEvaluator, pipelineModel, spark_test):
    predictions = pipelineModel.transform(spark_test)
    evaluator = MulticlassClassificationEvaluator(labelCol=Label_col,
    predictionCol="prediction", metricName="accuracy")
    random_forest_accuracy = evaluator.evaluate(predictions)
    return (random_forest_accuracy,)


@app.cell
def _(random_forest_accuracy):
    random_forest_accuracy
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
