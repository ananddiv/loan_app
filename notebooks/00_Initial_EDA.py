import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Initial EDA and Data Understanding**
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

    return SparkSession, mo, np, os, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **2. Initilaize Apache Spark Session**
    """)
    return


@app.cell
def _(SparkSession):
    spark = SparkSession.builder.appName("Loan_Application_Classifier").getOrCreate()
    return (spark,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **3. Setup file paths**
    """)
    return


@app.cell
def _(os):
    # Setup the base path
    base_dir = os.path.abspath('/Users/ananddivakaran/Documents/Anand/MDS/loan_app')
    print(f'Base Data Directory:{base_dir}')
    input_path = f"file://{os.path.join(base_dir, 'data/raw/loan_data.csv')}"
    print(f'Input file: {input_path}')
    output_path = os.path.join(base_dir, "/data/interim/loan_data_interim.csv")
    print(f'Output file:{output_path}')
    return input_path, output_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **4. Load the dataset**
    """)
    return


@app.cell
def _(input_path, spark):
    # Read data
    data = spark.read.csv(input_path, header=True, inferSchema=True)
    # Converting the data to pandas dataframe for easy analysis
    df = data.toPandas()
    df.head(5)
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Columns in the dataset
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **5. Checking for Null Values**
    """)
    return


@app.cell
def _(df, pd):
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
    missing_report = check_missing_data(df)
    pd.DataFrame(missing_report)
    return (check_missing_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### List of features with a NULL value
    """)
    return


@app.cell
def _(df):
    null_df = df.isnull().sum().reset_index()
    null_df.columns = ['Feature','Count']
    null_df[null_df['Count'] >0].reset_index()['Feature'].to_list()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **6. EDA and Analysis of features with a NULL value**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **6.1 Inference for Gender Feature**
    """)
    return


@app.cell
def _(df):

    df_gender_summ = df['Gender'].value_counts().reset_index()
    df_gender_summ
    return (df_gender_summ,)


@app.cell
def _(df_gender_summ, plt):
    bars_gender = plt.bar(df_gender_summ['Gender'], df_gender_summ['count'])

    # --- Adding Labels (Annotations) to each bar ---
    for bar_g in bars_gender:
        height_g = bar_g.get_height()
        # Add the text label above the bar, slightly offset from the top edge
        plt.text(
            bar_g.get_x() + bar_g.get_width() / 2, # X position (center of the bar)
            height_g - 25,                       # Y position (slightly above the bar height)
            f'{int(height_g)}',                  # The text to display (the count)
            ha='center',                       # Horizontal alignment: center
            va='bottom',color = 'white'        # Vertical alignment: bottom
        )

    plt.title('Gender Counts')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How many NULL values?
    """)
    return


@app.cell
def _(df):
    print(f'Number of records with NULL values in the Gender Feature:{df.Gender.isna().sum()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Observations: <br>  Number of males in the dataset if more than the females.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Imputation Stratergy:<br> Use the mode to fill the null values (do this only after the split)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **6.2 Inference for Marrried Feature**
    """)
    return


@app.cell
def _(df):
    df_married_summ = df['Married'].value_counts().reset_index()
    return (df_married_summ,)


@app.cell
def _(df_married_summ, plt):
    bars_married = plt.bar(df_married_summ['Married'], df_married_summ['count'])

    # --- Adding Labels (Annotations) to each bar ---
    for bar_m in bars_married:
        height_m = bar_m.get_height()
        # Add the text label above the bar, slightly offset from the top edge
        plt.text(
            bar_m.get_x() + bar_m.get_width() / 2, # X position (center of the bar)
            height_m - 25,                       # Y position (slightly above the bar height)
            f'{int(height_m)}',                  # The text to display (the count)
            ha='center',                       # Horizontal alignment: center
            va='bottom',color = 'white'        # Vertical alignment: bottom
        )

    plt.title('Married Counts')
    plt.xlabel('Married')
    plt.ylabel('Count')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### How many NULL's?
    """)
    return


@app.cell
def _(df):
    print(f'Number of records with NULL values in the Gender Feature:{df.Married.isna().sum()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Observations: <br> Number of Married people are more in the dataset than the unmarried.<br>
    The number of people with a NULL value in the "Married" feature is 3. <br>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Check if the records with NULL married feature has dependents
    """)
    return


@app.cell
def _(df):
    df_married_null = df[df['Married'].isnull()]
    return (df_married_null,)


@app.cell
def _(df_married_null):
    married_w_dep = df_married_null['Dependents'].value_counts()
    count_marrierd_w_dep = len(married_w_dep)
    print(f"Number of people with no martial status but with dependets:{count_marrierd_w_dep}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As there are no dependents for these records we can set the married flag to 'No'
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Imputation Stratergy:<br> If there are no dependents for the married people, set the married flag to "No"
    """)
    return


@app.cell
def _(df):
    df['Married'] = df['Married'].fillna("No")
    print(f'Number of records with NULL values in the Gender Feature:{df.Married.isna().sum()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **6.3 Inference for Dependents**
    """)
    return


@app.cell
def _(df, plt):
    df_dep_summ = df.Dependents.value_counts().reset_index()
    bars = plt.bar(df_dep_summ['Dependents'], df_dep_summ['count'])

    # --- Adding Labels (Annotations) to each bar ---
    for bar in bars:
        height = bar.get_height()
        # Add the text label above the bar, slightly offset from the top edge
        plt.text(
            bar.get_x() + bar.get_width() / 2, # X position (center of the bar)
            height - 20,                       # Y position (slightly above the bar height)
            f'{int(height)}',                  # The text to display (the count)
            ha='center',                       # Horizontal alignment: center
            va='bottom',color = 'white'        # Vertical alignment: bottom
        )

    plt.title('Dependents Counts')
    plt.xlabel('Dependents')
    plt.ylabel('Count')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### How many NULLS?
    """)
    return


@app.cell
def _(df):
    print(f'Number of records with NULL values in the Dependents Feature:{df.Dependents.isna().sum()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Observations: <br>
    There are only a small volume of data with null dependents - 15. We can set the dependents to zero for these 15 values as we could infer that not having a value means zero dependents.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Updating the records with NULL dependent count
    """)
    return


@app.cell
def _(df):
    df['Dependents'] = df['Dependents'].fillna(0)
    print(f'Number of records with NULL values in the Dependents Feature:{df.Dependents.isna().sum()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **6.4 Inference for Self_Employed Feature**
    """)
    return


@app.cell
def _(df, plt):
    df_selfe_summ = df.Self_Employed.value_counts().reset_index()
    bars_se = plt.bar(df_selfe_summ['Self_Employed'], df_selfe_summ['count'])

    # --- Adding Labels (Annotations) to each bar ---
    for bar_se in bars_se:
        height_se = bar_se.get_height()
        # Add the text label above the bar, slightly offset from the top edge
        plt.text(
            bar_se.get_x() + bar_se.get_width() / 2, # X position (center of the bar)
            height_se - 30,                       # Y position (slightly above the bar height)
            f'{int(height_se)}',                  # The text to display (the count)
            ha='center',                       # Horizontal alignment: center
            va='bottom',color = 'white' )       # Vertical alignment: bottom

    plt.title('Self Employed Counts')
    plt.xlabel('Self Emplyed')
    plt.ylabel('Count')
    plt.show()
    return (bar_se,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### How many NULLS?
    """)
    return


@app.cell
def _(df):
    print(f'Number of records with NULL values in the Self_Employed Feature:{df.Self_Employed.isna().sum()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Observations:
    There are only a small volume of data with null Self Employed - 32. We can set the the value of Self Employed to No for these 15 values as we could infer that not having a value not self employed.
    """)
    return


@app.cell
def _(df):
    df['Self_Employed'] = df['Self_Employed'].fillna('No')
    print(f'Number of records with NULL values in the Self_Employed Feature:{df.Self_Employed.isna().sum()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **6.5 Inference for Credit History Feature**
    """)
    return


@app.cell
def _(bar_se, df, plt):
    df_ch_summ = df.Credit_History.value_counts().reset_index()
    bars_ch = plt.bar(df_ch_summ['Credit_History'], df_ch_summ['count'])

    # --- Adding Labels (Annotations) to each bar ---
    for bar_ch in bars_ch:
        height_ch = bar_ch.get_height()
        # Add the text label above the bar, slightly offset from the top edge
        plt.text(
            bar_ch.get_x() + bar_se.get_width() / 2, # X position (center of the bar)
            height_ch - 30,                       # Y position (slightly above the bar height)
            f'{int(height_ch)}',                  # The text to display (the count)
            ha='center',                       # Horizontal alignment: center
            va='bottom',color = 'white' )       # Vertical alignment: bottom

    plt.title('Credit History Counts')
    plt.xlabel('Credit History')
    plt.ylabel('Count')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### How many NULLS?
    """)
    return


@app.cell
def _(df):
    print(f'Number of records with NULL values in the Credit History Feature:{df.Credit_History.isna().sum()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Observations: For the records with credit history as NULL - we can set the credit history as zero (No history) which is the same way.
    """)
    return


@app.cell
def _(df):
    df['Credit_History'] = df['Credit_History'].fillna(0)
    return


@app.cell
def _(df):
    print(f'Number of records with NULL values in the Credit History Feature:{df.Credit_History.isna().sum()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **6.6 Inference for Loan Amount Feature**
    """)
    return


@app.cell
def _(df, plt):
    df_la_summ = df.LoanAmount.value_counts().reset_index()
    bars_la = plt.bar(df_la_summ['LoanAmount'], df_la_summ['count'])

    # --- Adding Labels (Annotations) to each bar ---
    for bar_la in bars_la:
        height_la = bar_la.get_height()
        # Add the text label above the bar, slightly offset from the top edge
        plt.text(
            bar_la.get_x() + bar_la.get_width() / 2, # X position (center of the bar)
            height_la - 30,                       # Y position (slightly above the bar height)
            f'{int(height_la)}',                  # The text to display (the count)
            ha='center',                       # Horizontal alignment: center
            va='bottom',color = 'white' )       # Vertical alignment: bottom

    plt.title('Loan Amount Counts')
    plt.xlabel('Loan Amount')
    plt.ylabel('Count')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### How many NULLS?
    """)
    return


@app.cell
def _(df):
    print(f'Number of records with NULL values in the Loan Amount Feature:{df.LoanAmount.isna().sum()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Observations: <br> As loan Amount is a mandatory field that needs to be entered by the applicant - these records can be thought of an invalid records as it doesnt give any analytical value to aid in the prediction.
    """)
    return


@app.cell
def _(df):
    # Remove records from the dataframe with null loan amount
    df_cleaned = df.dropna(subset=['LoanAmount'])
    return (df_cleaned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **6.7 Inference for Loan Amount Term Feature**
    """)
    return


@app.cell
def _(df, plt):
    df_lat_summ = df.Loan_Amount_Term.value_counts().reset_index()
    bars_lat = plt.bar(df_lat_summ['Loan_Amount_Term'], df_lat_summ['count'])

    # --- Adding Labels (Annotations) to each bar ---
    for bar_lat in bars_lat:
        height_lat = bar_lat.get_height()
        # Add the text label above the bar, slightly offset from the top edge
        plt.text(
            bar_lat.get_x() + bar_lat.get_width() / 2, # X position (center of the bar)
            height_lat - 30,                       # Y position (slightly above the bar height)
            f'{int(height_lat)}',                  # The text to display (the count)
            ha='center',                       # Horizontal alignment: center
            va='bottom',color = 'white' )       # Vertical alignment: bottom

    plt.title('Loan Amount Term Counts')
    plt.xlabel('Loan Amount Term')
    plt.ylabel('Count')
    plt.show()
    return


@app.cell(hide_code=True)
def _(df):
    print(f'Number of records with NULL values in the Loan Amount Term Feature:{df.Loan_Amount_Term.isna().sum()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As these records has a loan Amount Vlaue, we can set the Loan Amount Term to the median loan Amount Term - This shoudl be done only after the split.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Disposition for NULL Values: <br>
    1. Gender - Use the mode to fill the null values (do this only after the split)
    2. Married - If the dependent count is 0, consider as un-married, else married. We can do this before the split.
    3. Dependents - If the dependent count is null - the medain value.(do this only after the split)
    4. Self_Employed - Set this to the median value. (do this after the split)
    5. Loan Amount - Remove the records with loan amount zero as this person might not need a loan itself.(Possible data error)
    6. Loan Amount Term - We can set the Loan Amount Term to the median loan Amount Term - (This should be done only after the split)
    7. Credit History - Set as No-credit history. (Initilize this to no credit history as it is not reported by the credit beauroes.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How many NULLS after the excercise?
    """)
    return


@app.cell
def _(check_missing_data, df_cleaned, pd):
    # Run the check
    missing_report_lat = check_missing_data(df_cleaned)
    pd.DataFrame(missing_report_lat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **7. Check for Outliers in Numeric Fields and the Disposition**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **7.1 Identify the numeric features**
    """)
    return


@app.cell
def _(df, np):
    numeric_features = df.select_dtypes(include = np.number).columns.tolist()
    numeric_features
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define a function to check for outliers
    """)
    return


@app.function
def detect_iqr(df_dat, feature):
    Q1 = df_dat[feature].quantile(0.25)
    Q3 = df_dat[feature].quantile(0.75)
    IQR = Q3-Q1
    Lower_Bound = Q1 - 1.5*IQR
    Upper_Bound = Q3 + 1.5*IQR
    Outliers = df_dat[(df_dat[feature] < Lower_Bound) | (df_dat[feature] > Upper_Bound)]
    print(f"\n--- IQR Method for {feature} ---")
    print(f"Lower Bound: {Lower_Bound:.2f}")
    print(f"Upper Bound: {Upper_Bound:.2f}")
    print(f"Number of Outliers Detected: {len(Outliers)}")
    return Lower_Bound, Upper_Bound,Outliers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **7.1 Check for outliers in Applicant Income Feature**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot the data using a regular box plot to understand the outliers.
    """)
    return


@app.cell
def _(df_cleaned, plt, sns):
    sns.boxplot(data=df_cleaned, y = 'ApplicantIncome')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are a few records with very high Income Amounts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Understand the distribution of outliers using histplot.
    """)
    return


@app.cell
def _(df_cleaned, plt, sns):
    plt.figure(figsize=(10,6))
    sns.histplot(df_cleaned.ApplicantIncome, kde=True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #####Run the outlier Detection
    """)
    return


@app.cell
def _(df_cleaned):
    ## Run the detection
    lower_bound, upper_bound, iqr_outliers = detect_iqr(df_cleaned, 'ApplicantIncome')
    return iqr_outliers, lower_bound, upper_bound


@app.cell
def _(iqr_outliers, lower_bound, upper_bound):
    print(f'Number of outliers below lower bound: {iqr_outliers[iqr_outliers.ApplicantIncome < lower_bound].shape[0]}')
    print(f'Number of outliers above upper bound: {iqr_outliers[iqr_outliers.ApplicantIncome > upper_bound].shape[0]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Income data is extremely right sqewed and if we use linear models, the outliers could impact the coefficents. Apart from this, the relationship between income and probability to default is not linear - it is step-wise. Hence we will discretize the Applicant Income feature. <br>

    Low Income: < 30,000 <br>
    Moderate Income: 30,001 - 75,000 <br>
    High Income: 75,001 - 150,000 <br>
    Very High Income: > 150,001
    """)
    return


@app.cell
def _(df_cleaned):
    # Create a function to perfrom the binning.
    def bucketize_income (income):
        if income <= 30000:
            return 'Low Income'
        elif income > 30000 and income <= 75000:
            return 'Medium Income'
        elif income > 75000 and income <= 150000:
            return 'High Income'
        elif income > 150000:
            return 'Very High Income'

    df_cleaned['ApplicantIncome'] = df_cleaned['ApplicantIncome'].apply(bucketize_income)
    df_cleaned['ApplicantIncome'].value_counts()
    return (bucketize_income,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **7.2 Check for outliers in CoApplicantIncome Feature**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot the data using a regular box plot to understand the outliers.
    """)
    return


@app.cell
def _(df_cleaned, plt, sns):
    sns.boxplot(data=df_cleaned, y = 'CoapplicantIncome')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Understand the distribution of outliers using histplot.
    """)
    return


@app.cell
def _(df_cleaned, plt, sns):
    plt.figure(figsize=(10,6))
    sns.histplot(df_cleaned.CoapplicantIncome, kde=True)
    plt.show()
    return


@app.cell
def _(df_cleaned):
    ## Run the detection
    lb_caincome, ub_caincome, iqr_outliers_caincome = detect_iqr(df_cleaned, 'CoapplicantIncome')
    return iqr_outliers_caincome, lb_caincome, ub_caincome


@app.cell
def _(iqr_outliers_caincome, lb_caincome, ub_caincome):
    print(f'Number of outliers below lower bound: {iqr_outliers_caincome[iqr_outliers_caincome.CoapplicantIncome < lb_caincome].shape[0]}')
    print(f'Number of outliers above upper bound: {iqr_outliers_caincome[iqr_outliers_caincome.CoapplicantIncome > ub_caincome].shape[0]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Co Applicant Income data is extremely right sqewed and if we use linear models, the outliers could impact the coefficents. Apart from this, the relationship between income and probability to default is not linear - it is step-wise. Hence we will discretize the Applicant Income feature.
    """)
    return


@app.cell
def _(bucketize_income, df_cleaned):
    df_cleaned['CoapplicantIncome'] = df_cleaned['CoapplicantIncome'].apply(bucketize_income)
    df_cleaned['CoapplicantIncome'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **7.3 Check for outliers in Loan Amount Feature**
    """)
    return


@app.cell
def _(df_cleaned, plt, sns):
    sns.boxplot(data=df_cleaned, y = 'LoanAmount')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Understand the distribution of outliers using histplot.
    """)
    return


@app.cell
def _(df_cleaned, plt, sns):
    plt.figure(figsize=(10,6))
    sns.histplot(df_cleaned.LoanAmount, kde=True)
    plt.show()
    return


@app.cell
def _(df_cleaned):
    ## Run the detection
    lb_la, ub_la, iqr_outliers_la = detect_iqr(df_cleaned, 'LoanAmount')
    return iqr_outliers_la, lb_la, ub_la


@app.cell
def _(iqr_outliers_la, lb_la, ub_la):
    print(f'Number of outliers below lower bound: {iqr_outliers_la[iqr_outliers_la.LoanAmount < lb_la].shape[0]}')
    print(f'Number of outliers above upper bound: {iqr_outliers_la[iqr_outliers_la.LoanAmount > ub_la].shape[0]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### We will do a quantile binning for the loan amount. But we will perfrom quantile billing onlu after we split the data to avoid data leakage.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **7.4 Check for outliers in Loan Amount Term Feature**
    """)
    return


@app.cell
def _(df_cleaned, plt, sns):
    sns.boxplot(data=df_cleaned, y = 'Loan_Amount_Term')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Understand the distribution of outliers using histplot.
    """)
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(10,6))
    sns.histplot(df.Loan_Amount_Term, kde=True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### The data looks descrete and hence we can use the time based bucketing to group the loan terms into standard buckets. <br>
    0-2 Years: Short Term <br>
    2-5 Years: Medium Term <br>
    5-10 Years: Long Term <br>
    greater than 10 Years: Very Long Term <br>
    """)
    return


@app.cell
def _(df_cleaned):
    # Create a function to perfrom the binning.
    def bucket_loan_term (loan_term):
        if loan_term <= 24:
            return 'Short Term'
        elif loan_term > 24 and loan_term <= 60:
            return 'Medium Term'
        elif loan_term > 60 and loan_term <= 120:
            return 'Long Term'
        elif loan_term > 120:
            return 'Very Long Term'

    df_cleaned['Loan_Amount_Term'] = df_cleaned['Loan_Amount_Term'].apply(bucket_loan_term)
    df_cleaned['Loan_Amount_Term'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **7.5 Check for outliers in Credit History Feature**
    """)
    return


@app.cell
def _(df_cleaned, plt, sns):
    sns.boxplot(data=df_cleaned, y = 'Credit_History')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### We do not see any outliers for this feature.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Save the data into an intermediate file
    """)
    return


@app.cell
def _(output_path):
    print(f'Saving the data in:{output_path}')
    return


@app.cell
def _(df_cleaned, spark):
    # Save the cleaned dataframe into a csv file. 
    df_cleaned.to_csv("/Users/ananddivakaran/Documents/Anand/MDS/loan_app/data/interim/loan_data_interim.csv", index=False, header=True)
    # Stop the Spark Session when done
    spark.stop()
    return


if __name__ == "__main__":
    app.run()
