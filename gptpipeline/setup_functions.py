import os
import pandas as pd
import glob

def generate_primary_csv(dest_csv_path, csv_file_name, folder_path, **default_features):
    """
    Generates a CSV file with specified default features and file paths of text files in a given folder.

    This function first checks if a CSV file with the given name exists in the specified destination path.
    If it exists, the function returns False and does nothing. Otherwise, it creates a new CSV file with 
    columns based on the provided default feature names and their default values, and an additional 
    'File Path' column. It then populates the DataFrame with the paths of all .txt and .text files in the 
    specified folder and the default values for other features, and finally writes the DataFrame to a CSV file.

    Args:
        dest_csv_path (str): The destination path where the CSV file will be saved.
        csv_file_name (str): The name of the CSV file to be created.
        folder_path (str): The path to the folder from which .txt and .text file paths will be read.
        **default_features: Variable length dictionary of DataFrame feature names with their default values.
                            The values can be of any type that can be stored in CSV format.

    Returns:
        bool: True if a new CSV file was created, False if the file already exists.

    Example:
        generate_primary_csv('/path/to/destination', 'data.csv', '/path/to/folder', 
                             feature1='default1', feature2='default2')
    """

    # Construct the full path for the csv file
    full_csv_path = os.path.join(dest_csv_path, csv_file_name)
    
    # Check if the CSV file already exists
    if os.path.exists(full_csv_path):
        return False

    # Initialize the DataFrame with 'File Path' and 'Complete' columns first
    columns = ['File Path', 'Complete'] + list(default_features.keys())
    df = pd.DataFrame(columns=columns)

    # Identify all .txt or .text files in the folder
    rows_to_add = []
    for file_path in glob.glob(f"{folder_path}/*.txt") + glob.glob(f"{folder_path}/*.text"):
        # Create a new row with default values, setting 'File Path' and 'Complete'
        new_row = {'File Path': file_path, 'Complete': 0}
        new_row.update(default_features)
        rows_to_add.append(new_row)

    # If there are no files, return False
    if not rows_to_add:
        return False

    # Concatenate the new rows to the DataFrame
    df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)

    # Write DataFrame to CSV
    df.to_csv(full_csv_path, index=False)

    return True    