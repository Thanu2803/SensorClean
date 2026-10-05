import pandas as pd


# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Clean the dataset
def clean_data(df):

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Find numerical columns
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    # Fill missing values with median
    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    return df


# Main program
if __name__ == "__main__":

    # Load data
    df = load_data("data/sensor_data.csv")

    print("===== ORIGINAL DATA =====")
    print(df.head())

    print("\n===== DATASET INFORMATION =====")
    print(df.info())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DUPLICATE ROWS =====")
    print(df.duplicated().sum())

    # Clean data
    df = clean_data(df)

    print("\n===== AFTER CLEANING =====")
    print("Missing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\n===== STATISTICS =====")
    print(df.describe())