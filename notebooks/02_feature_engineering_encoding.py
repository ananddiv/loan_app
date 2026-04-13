import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Feature Engineering**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Import Python Libraries
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

    return SparkSession, mo, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Initialize the spark Session
    """)
    return


@app.cell
def _(SparkSession):
    spark = SparkSession.builder.appName("Loan_Application_Classifier").getOrCreate()
    return (spark,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Setup the file paths
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
    ## 4. Import Files
    """)
    return


@app.cell
def _(output_path_test, output_path_train, spark):
    df_train = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(output_path_train)

    df_test = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(output_path_test)
    return df_test, df_train


@app.cell
def _(df_train):
    df_train.count()
    return


@app.cell
def _(df_test):
    df_test.count()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
