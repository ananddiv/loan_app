import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Splitting the data into Train and Test Datasets
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **1. Import Python Libraries**
    """)
    return


@app.cell
def _():
    import marimo as mo
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

    return SparkSession, mo, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **2. Initialize Apache Spark Session**
    """)
    return


@app.cell
def _(SparkSession):
    spark = SparkSession.builder.appName("Loan_Application_Classifier").getOrCreate()
    return (spark,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **3. Setup File Paths**
    """)
    return


@app.cell
def _(os):
    # Setup the base path
    base_dir = os.path.abspath('/Users/ananddivakaran/Documents/Anand/MDS/loan_app')
    print(f'Base Data Directory:{base_dir}')
    input_path = f"file://{os.path.join(base_dir,'data/interim/loan_data_interim.csv')}"
    print(f'Input file: {input_path}')
    output_path_train = os.path.join(base_dir, "data/processed/loan_data_train")
    print(f'Train file:{output_path_train}')
    output_path_test = os.path.join(base_dir, "data/processed/loan_data_test")
    print(f'Test file:{output_path_test}')
    return input_path, output_path_test, output_path_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **4. Load the dataset**
    """)
    return


@app.cell
def _(input_path, spark):
    # Read data
    data_processed = spark.read.csv(input_path, header=True, inferSchema=True)
    # Converting the data to pandas dataframe for easy analysis
    df = data_processed.toPandas()
    df.head(5)
    return (data_processed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **5. Split the data**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Splitting the data into Train and Test Data with 70:30 ratio.
    """)
    return


@app.cell
def _(data_processed):
    train_data, test_data = data_processed.randomSplit([0.7, 0.3], seed=42)
    train_data_df = train_data.toPandas()
    test_data_df = test_data.toPandas()
    return test_data, test_data_df, train_data, train_data_df


@app.cell
def _(train_data):
    print(f'Count of Train Dataset: {train_data.count()}')
    return


@app.cell
def _(test_data):
    print(f'Count of Train Dataset: {test_data.count()}')
    return


@app.cell
def _(train_data_df):
    train_data_df.info()
    return


@app.cell
def _(test_data_df, train_data_df):
    train_data_df['Credit_History'] = train_data_df['Credit_History'].astype('category')
    test_data_df['Credit_History'] = test_data_df['Credit_History'].astype('category')
    return


@app.cell
def _(train_data_df):
    train_data_df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **6. Handle NULLS after the split (Imputation)**
    """)
    return


@app.cell
def _(pd, train_data_df):
    def check_missing_data(df):
        report = []

        for col in df.columns:
            # 1. Count actual Nulls (NaN)
            null_count = df[col].isnull().sum()
            total_rows = len(df)
            null_percent = (null_count / total_rows) * 100

            report.append({
                'Column Name': col,
                'Null Count': null_count,
                'Null %': round(null_percent, 2)
            })

        return pd.DataFrame(report).sort_values('Null Count', ascending=False)

    # Run the check
    missing_report = check_missing_data(train_data_df)
    pd.DataFrame(missing_report)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nulls are present in the below features <br>
    1. Loan Amount Term - Fill it with the mode value
    2. Gender - Use the mode to fill the null
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 Handle NULLs in the feture - Loan Amount Term
    """)
    return


@app.cell
def _(test_data_df, train_data_df):
    # Find the mode from the train dataset
    median_loan_term = train_data_df.Loan_Amount_Term.mode()
    # Fill the nulls with mode in train dataset
    train_data_df['Loan_Amount_Term'] = train_data_df['Loan_Amount_Term'].fillna(median_loan_term[0])
    # Fill the nulls with mode in test dataset - avoid data leakage
    test_data_df['Loan_Amount_Term'] = test_data_df['Loan_Amount_Term'].fillna(median_loan_term[0])
    return


@app.cell
def _(train_data_df):
    train_data_df.isnull().sum()
    return


@app.cell
def _(test_data_df):
    test_data_df.isnull().sum()
    return


@app.cell
def _(train_data_df):
    train_data_df.Loan_Amount_Term.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Handle NULLs in the feture - Gender
    """)
    return


@app.cell
def _(test_data_df, train_data_df):
    # Find the mode from the train dataset
    median_gender = train_data_df.Gender.mode()
    # Fill the nulls with mode in train dataset
    train_data_df['Gender'] = train_data_df['Gender'].fillna(median_gender[0])
    # Fill the nulls with mode in test dataset - avoid data leakage
    test_data_df['Gender'] = test_data_df['Gender'].fillna(median_gender[0])
    return


@app.cell
def _(train_data_df):
    train_data_df.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **7. Saving the data in the folder**
    """)
    return


@app.cell
def _(output_path_test, output_path_train, test_data, train_data):
    train_data.write.option("header", "true") \
            .option("sep", ",") \
            .mode("overwrite") \
            .csv(output_path_train)

    test_data.write.option("header", "true") \
            .option("sep", ",") \
            .mode("overwrite") \
            .csv(output_path_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **8. Stop the spark session**
    """)
    return


@app.cell
def _(spark):
    spark.stop()
    return


if __name__ == "__main__":
    app.run()
