# =========================
# File Handling and OS
# =========================
import os
import zipfile
import gdown

# =========================
# Miscellaneous
# =========================
import time
import warnings
import config

# =========================
# Data Processing
# =========================
import numpy as np
import pandas as pd
import idx2numpy
from sklearn.impute import SimpleImputer

# =========================
# Image Processing
# =========================
import cv2
from scipy.ndimage import gaussian_filter, sobel
from scipy.stats import boxcox


# =========================
# Visualization
# =========================
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# =========================
# Machine Learning
# =========================
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import (
    MinMaxScaler, 
    OneHotEncoder, 
    OrdinalEncoder, 
    PowerTransformer, 
    StandardScaler
)
from xgboost import XGBClassifier

# =========================
# Deep Learning
# =========================
import tensorflow as tf
from tensorflow.keras import models, layers # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

def AA02_check_unique_values(dataframe):
    """
    Calculate the number of unique values, total values,
    and percentage of unique values for each column in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A summary DataFrame with unique value statistics.
    """
    # Calculate unique values, total values, and percentage of unique values
    unique_counts = dataframe.nunique()
    total_counts = dataframe.count()
    percentages = (unique_counts / total_counts) * 100

    # Combine the results into a DataFrame for better AA02_display
    summary_AA02_df = pd.DataFrame({
        'Unique Values': unique_counts,
        'Total Values': total_counts,
        'Percentage (%)': percentages
    })

    return summary_AA02_df
    

# Function to calculate missing data information
def AA02_missing_data_info(AA02_sample_data):
    """
    Calculate the missing count and percentage for each variable in the DataFrame.
    Args:
        AA02_sample_data (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: A DataFrame with missing data information
    """
    # Calculate missing count and percentage
    AA02_missing_count = AA02_sample_data.isnull().sum()
    AA02_missing_percentage = (AA02_missing_count / len(AA02_sample_data)) * 100

    # Create a DataFrame with missing data information
    AA02_missing_info = pd.DataFrame({
        'AA02_Variable': AA02_sample_data.columns,
        'AA02_Missing_Count': AA02_missing_count.values,
        'AA02_Missing_Percentage': AA02_missing_percentage.values
    }).reset_index(drop=True)

    # Format the percentage column
    AA02_missing_info['AA02_Missing_Percentage'] = AA02_missing_info['AA02_Missing_Percentage'].round(2).astype(str) + '%'

    return AA02_missing_info


def AA02_display_full_dataframe(df):
    """
    Display the entire DataFrame without truncation.

    Args:
        df (pd.DataFrame): The DataFrame to display.

    Returns:
        None
    """
    # Set display options for max columns and rows
    pd.set_option('display.max_columns', None)

    # Display the DataFrame
    display(df)

    # Reset options to defaults after displaying
    pd.reset_option('display.max_columns')

def AA02_display_all(dataframes):
    """
    Display all DataFrames in the dictionary.

    Args:
        dataframes (dict): A dictionary where keys are DataFrame names and values are DataFrames.

    Returns:
        None
    """
    for name, df in dataframes.items():
        print(f"DataFrame: {name}")
        AA02_display_full_dataframe(df)


# Function to omit variables with more than a threshold of missing values and log omitted variables
def AA02_clean_data_with_logging(
    AA02_sample_data,
    AA02_categorical_columns,
    AA02_non_categorical_columns,
    AA02_columns,
    missing_threshold=50
):
    """
    Log the initial state of the data.

    Args:
        AA02_sample_data (pd.DataFrame): The input DataFrame.

    Returns:
        None
    """
    # Log the initial state of the data
    print("Initial Data State:")
    AA02_display_full_dataframe(AA02_sample_data)

    # Calculate missing percentage for each variable
    AA02_missing_percentage = (AA02_sample_data.isnull().sum() / len(AA02_sample_data)) * 100

    # Identify variables to omit (missing percentage > threshold)
    variables_to_omit = AA02_missing_percentage[AA02_missing_percentage > missing_threshold]

    # Create a DataFrame for omitted variables
    omitted_info = []
    for variable, percentage in variables_to_omit.items():
        if variable in AA02_categorical_columns:
            source = "AA02_categorical_columns"
        elif variable in AA02_non_categorical_columns:
            source = "AA02_non_categorical_columns"
        elif variable in AA02_columns:
            source = "AA02_columns"
        else:
            source = "Unknown"

        omitted_info.append({
            "Variable": variable,
            "Missing_Percentage": round(percentage, 2),
            "Omitted_From": source
        })

    # Convert omitted info to DataFrame
    AA02_omitted_df = pd.DataFrame(omitted_info)

    # Identify variables to keep
    variables_to_keep = AA02_missing_percentage[AA02_missing_percentage <= missing_threshold].index.tolist()

    # Filter the dataset
    AA02_sample_data_cleaned = AA02_sample_data[variables_to_keep]

    # Update the lists (only keep variables that are not omitted)
    AA02_columns[:] = [col for col in AA02_columns if col in variables_to_keep]
    AA02_categorical_columns[:] = [col for col in AA02_categorical_columns if col in variables_to_keep]
    AA02_non_categorical_columns[:] = [col for col in AA02_non_categorical_columns if col in variables_to_keep]

    # Print the DataFrame of omitted variables
    print("Variables Omitted Due to Missing Values (> {}%):".format(missing_threshold))
    AA02_display_full_dataframe(AA02_omitted_df)

    return AA02_sample_data_cleaned

def AA02_remove_records_with_missing_values(AA02_sample_data_dropped_variable, percentage):
    """
    This function removes records from the DataFrame where the percentage of missing values 
    exceeds the specified threshold.

    Parameters:
        AA02_sample_data_dropped_variable (pd.DataFrame): The input DataFrame.
        percentage (float): The threshold percentage of missing values.

    Returns:
        pd.DataFrame: Modified DataFrame with records removed.
    """
    # Calculate the threshold for missing values based on the given percentage
    threshold = (percentage / 100) * AA02_sample_data_dropped_variable.shape[1]

    # Identify records with missing values exceeding the threshold
    AA02_records_with_excessive_missing = AA02_sample_data_dropped_variable[AA02_sample_data_dropped_variable.isnull().sum(axis=1) > threshold]

    # Print the records with excessive missing values
    print("Records with more than", percentage, "% missing values:")
    AA02_records_with_excessive_missing

    # Remove those records from the original DataFrame
    AA02_sample_data_dropped_records = AA02_sample_data_dropped_variable.drop(index=AA02_records_with_excessive_missing.index)

    # Return the modified DataFrame
    return AA02_sample_data_dropped_records, AA02_records_with_excessive_missing

def AA02_impute_columns_with_mean_or_median(AA02_df, columns):
    """
    Impute missing values in specified columns with mean or median based on significant difference.

    Parameters:
    AA02_df (pd.DataFrame): The dataframe to impute.
    columns (list): List of column names to impute.

    Returns:
    pd.DataFrame: The updated dataframe after imputation.
    pd.DataFrame: A dataframe with imputation details for each column.
    """
    imputation_details = []

    for col in columns:
        # Ensure column is numeric
        AA02_df[col] = pd.to_numeric(AA02_df[col], errors='coerce')

        # Replace invalid values with NaN
        AA02_df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Skip if column has no missing values
        if AA02_df[col].isnull().sum() == 0:
            imputation_details.append({
                'Variable': col,
                'Imputation Method': 'None (No Missing Values)',
                'Significant Difference': 0,
                'Percentage Difference': 0.00
            })
            continue

        # Calculate mean and median
        AA02_col_mean = AA02_df[col].mean()
        AA02_col_median = AA02_df[col].median()

        # Calculate percentage difference
        percentage_diff = abs(AA02_col_mean - AA02_col_median) / max(abs(AA02_col_mean), abs(AA02_col_median)) * 100
        significant_diff = int(percentage_diff > 10)  # Binary: 1 if significant, 0 otherwise

        # Choose strategy based on significant difference
        if significant_diff:
            imputation_method = 'Median'
            AA02_imputer = SimpleImputer(strategy='median')
        else:
            imputation_method = 'Mean'
            AA02_imputer = SimpleImputer(strategy='mean')

        # Apply the AA02_imputer
        AA02_df[[col]] = AA02_imputer.fit_transform(AA02_df[[col]])

        # Append details to the list
        imputation_details.append({
            'Variable': col,
            'Imputation Method': imputation_method,
            'Significant Difference': significant_diff,
            'Percentage Difference': round(percentage_diff, 2)
        })

    # Create a dataframe with imputation details
    imputation_details_df = pd.DataFrame(imputation_details)

    return AA02_df, imputation_details_df

def AA02_convert_to_numeric(dataframe):
    """
    Converts all columns in the DataFrame into numeric types where possible.
    - Strings will be converted to numeric if feasible.
    - True/False (case insensitive) will be converted to 1 and 0.
    - If a value cannot be converted to numeric, it will remain as is.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame to convert.

    Returns:
        pd.DataFrame: A DataFrame with numeric conversions applied where possible.
    """
    def AA02_safe_convert(value):
        # Handle case-insensitive True/False
        if isinstance(value, str):
            if value.strip().lower() == 'true':
                return 1
            elif value.strip().lower() == 'false':
                return 0
        
        # Try to convert other values to numeric
        try:
            return pd.to_numeric(value, errors='raise')
        except:
            return value

    # Apply safe conversion to all elements in the DataFrame
    dataframe = dataframe.applymap(AA02_safe_convert)

    return dataframe


def AA02_encode_categorical_columns(dataframe, ordinal_columns, nominal_columns, use_one_hot_for_nominal=False, ordinal_categories=None):
    """
    Encodes categorical columns in the dataframe using either OrdinalEncoder or OneHotEncoder for nominal columns
    and OrdinalEncoder for ordinal columns.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame to encode.
        ordinal_columns (list): List of ordinal column names to encode.
        nominal_columns (list): List of nominal column names to encode.
        use_one_hot_for_nominal (bool): If True, use OneHotEncoder for nominal columns. Otherwise, use OrdinalEncoder.
        ordinal_categories (list of lists): The order of categories for ordinal columns. Pass None if not applicable.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
    """
    # Make a copy of the DataFrame
    dataframe_encoded = dataframe.copy()

    # Initialize OrdinalEncoder for ordinal columns with specified order
    if ordinal_categories:
        ordinal_encoder_ordinal = OrdinalEncoder(categories=ordinal_categories)
    else:
        ordinal_encoder_ordinal = OrdinalEncoder()

    # Encode ordinal columns
    if ordinal_columns:
        dataframe_encoded[ordinal_columns] = ordinal_encoder_ordinal.fit_transform(
            dataframe_encoded[ordinal_columns].astype(str)
        )

    # Encode nominal columns
    if use_one_hot_for_nominal:
        # Exclude numeric columns from one-hot encoding
        nominal_columns_to_encode = [col for col in nominal_columns if not pd.api.types.is_numeric_dtype(dataframe[col])]
        if nominal_columns_to_encode:
            one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_nominal_columns = one_hot_encoder.fit_transform(dataframe_encoded[nominal_columns_to_encode].astype(str))
            encoded_nominal_df = pd.DataFrame(
                encoded_nominal_columns,
                columns=one_hot_encoder.get_feature_names_out(nominal_columns_to_encode),
                index=dataframe_encoded.index
            )
            dataframe_encoded = dataframe_encoded.drop(nominal_columns_to_encode, axis=1)
            dataframe_encoded = pd.concat([dataframe_encoded, encoded_nominal_df], axis=1)
    else:
        ordinal_encoder_nominal = OrdinalEncoder()
        dataframe_encoded[nominal_columns] = ordinal_encoder_nominal.fit_transform(
            dataframe_encoded[nominal_columns].astype(str)
        )

    return dataframe_encoded

# Function to handle transformations based on distribution characteristics
def AA02_apply_transformations(AA02_sample_data, columns):
    # Initialize a list to store transformation logs
    AA02_transformation_logs = []

    for column in columns:
        # Compute AA02_skewness and AA02_kurtosis
        AA02_skewness = AA02_sample_data[column].skew()
        AA02_kurtosis = AA02_sample_data[column].kurt()
        AA02_action = "None"  # Default AA02_action

        # Handle Right Skew (Positive Skew)
        if AA02_skewness > 1:
            AA02_action = "Log Transformation"
            AA02_sample_data[column] = np.log1p(AA02_sample_data[column])

        # Handle Left Skew (Negative Skew)
        elif AA02_skewness < -1:
            AA02_action = "Reflect and Log Transformation"
            AA02_sample_data[column] = np.log1p(AA02_sample_data[column].max() - AA02_sample_data[column])

        # Handle High Kurtosis (Heavy Tails)
        if AA02_kurtosis > 3:
            try:
                AA02_action = "Box-Cox Transformation"
                AA02_sample_data[column], _ = boxcox(AA02_sample_data[column].clip(lower=1))
            except ValueError:
                AA02_action = "Box-Cox Failed, Applied Yeo-Johnson"
                transformer = PowerTransformer(method='yeo-johnson')
                AA02_sample_data[column] = transformer.fit_transform(AA02_sample_data[[column]])

        # Handle Low Kurtosis (Light Tails)
        elif AA02_kurtosis < 3 and AA02_action == "None":
            AA02_action = "Yeo-Johnson Transformation"
            transformer = PowerTransformer(method='yeo-johnson')
            AA02_sample_data[column] = transformer.fit_transform(AA02_sample_data[[column]])

        AA02_skewness_after_transformation = AA02_sample_data[column].skew()
        AA02_kurtosis_after_transformation = AA02_sample_data[column].kurt()

        # Append the log entry
        AA02_transformation_logs.append({
            'Column Name': column,
            'Skewness Before Transformation': AA02_skewness,
            'Kurtosis Before Transformationv': AA02_kurtosis,
            'Action Taken': AA02_action,
            'Skewness After Transformation': AA02_skewness_after_transformation,
            'Kurtosis After Transformationv': AA02_kurtosis_after_transformation
        })

    # Create a DataFrame for transformation logs
    transformation_log_AA02_df = pd.DataFrame(AA02_transformation_logs)
    return AA02_sample_data, transformation_log_AA02_df


def AA02_scale_dataframe(AA02_df,AA02_y_columns, method='standard'):
    """
    Scales numeric columns of the input DataFrame, excluding binary columns.

    Parameters:
        AA02_df (pd.DataFrame): Input DataFrame to scale.
        method (str): Scaling method, either 'standard' (default) for StandardScaler or 'minmax' for MinMaxScaler.

    Returns:
        pd.DataFrame: Scaled DataFrame with the same column names as the input.
    """
    if not isinstance(AA02_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Select numeric columns only
    AA02_numeric_cols = AA02_df.select_dtypes(include=['float64', 'int64']).columns

    # Exclude binary columns (those with only two unique values)
    AA02_non_binary_cols = [col for col in AA02_numeric_cols if AA02_df[col].nunique() > 2]

    AA02_non_binary_cols = [
    var for var in AA02_non_binary_cols if var not in AA02_y_columns
]

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid method. Use 'standard' or 'minmax'.")

    # Scale non-binary numeric columns
    AA02_df_scaled = AA02_df.copy()
    AA02_df_scaled[AA02_non_binary_cols] = scaler.fit_transform(AA02_df[AA02_non_binary_cols])

    return AA02_df_scaled

