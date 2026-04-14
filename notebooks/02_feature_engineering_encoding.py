import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Feature Engineering & Statistical Significance**
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
    from scipy.stats import chi2_contingency

    import os
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import StringIndexer, VectorAssembler
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline

    return SparkSession, chi2_contingency, mo, os, pd, plt, sns


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
def _(df_train):
    df_train_pd = df_train.toPandas()
    df_train_pd.info()
    return (df_train_pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Select categorial variables to check for statistical significance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1 Identify the categorical variables
    """)
    return


@app.cell
def _(df_train_pd):
    categorical_features = df_train_pd.select_dtypes(include=['category','object','str']).columns.tolist()
    return (categorical_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Remove featues of no analytical importance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Removing Loan_ID as it is used to identify a row and is of no analytical value.
    """)
    return


@app.cell
def _(categorical_features):
    index_0 = categorical_features.index("Loan_ID")
    print(f'Loan_ID is in index:{index_0}')
    categorical_features.pop(index_0)
    categorical_features
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 Remove the target variable
    """)
    return


@app.cell
def _(categorical_features):
    index_1 = categorical_features.index("Loan_Status")
    print(f'Loan_Status is in index:{index_1}')
    categorical_features.pop(index_1)
    categorical_features
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6. Check for statistical significance (Chi-Square Test) of categorical variables
    """)
    return


@app.cell
def _(categorical_features, chi2_contingency, df_train_pd, pd):
    # Define categorical variables to analyze
    #categorical_cols = ['Sex', 'Race', 'Location', 'Body_Part', 'Diagnosis','Product_1','Other_Race','Hispanic','Fire_Involvement','True_Alcohol','True_Drug','Stratum','Treatment_Month']

    predator_features = []
    # Loop through and plot the "Risk"
    for col in categorical_features:
        # Example: Check if Sex predicts Hospitalization
        contingency_table = pd.crosstab(df_train_pd[col], df_train_pd['Loan_Status'])
        #print(contingency_table)
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        if p < 0.05:
            print(f'{col} is a significant predictor. pvalue {p}')
            predator_features.append(col)
        else:
            print(f'{col} is likely noise.pvalue{p}')

    print(f'Predator Features:{predator_features}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7. Check for the statistical significance of numeric variables
    """)
    return


@app.cell
def _(df_train_pd, plt, sns):
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Violin Plot
    sns.violinplot(data=df_train_pd, x='Loan_Status', y='LoanAmount', ax=axes[0], hue='Loan_Status')
    axes[0].set_title('Violin Plot: Loan Amount vs. Loan Approved')
    axes[0].set_ylabel('Loan Amount ($)')
    axes[0].set_xlabel('Loan Approved')

    # 2. Box Plot 
    sns.boxplot(data=df_train_pd, x='Loan_Status', y='LoanAmount', ax=axes[1], hue='Loan_Status')
    axes[1].set_title('Box Plot: Loan Amount vs. Loan Approved')
    axes[1].set_ylabel('Loan Amount ($)')
    axes[1].set_xlabel('Loan Approved')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The median loan amount for the data approval Yes is similar to the mdeian loan amount for No. There doesnt seem to be any predictive power for the loan amount.
    """)
    return


if __name__ == "__main__":
    app.run()
