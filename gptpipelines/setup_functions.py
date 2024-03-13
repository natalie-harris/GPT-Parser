import os
import pandas as pd
import glob

def generate_primary_csv(folder_path, dest_csv_path=None, csv_file_name='files.csv', default_features={}):
    """
    Generates a CSV file listing files in a folder to be used as the first df in a GPTPipeline.

    This function searches for text files (`.txt` and `.text`) in a specified folder, compiles their
    file paths, and generates a CSV file with these paths and additional default features. If a CSV file
    with the specified name already exists at the destination path, the function returns `False` without
    creating a new file. Otherwise, it creates a new CSV file, populating it with the default feature values
    and the file paths of the text files found.

    Parameters
    ----------
    folder_path : str
        The path to the folder from which .txt and .text file paths will be collected.
    dest_csv_path : str, optional
        The destination path where the CSV file will be saved. If `None` (default), uses `folder_path`.
    csv_file_name : str, optional
        The name of the CSV file to be created (default is 'files.csv').
    default_features : dict, optional
        A dictionary specifying the default features and their values to include in the CSV file. Each key-value
        pair corresponds to a column name and its default value (default is an empty dict).

    Returns
    -------
    bool
        `True` if a new CSV file was successfully created; `False` if the CSV file already exists at the
        specified destination path.

    Examples
    --------
    Create a CSV file 'data.csv' in '/path/to/destination' directory listing all text files from 
    '/path/to/folder', with additional columns 'feature1' and 'feature2' having default values 'default1' 
    and 'default2', respectively:

    >>> generate_primary_csv('/path/to/folder', '/path/to/destination', 'data.csv', 
                             default_features={'feature1': 'default1', 'feature2': 'default2'})
    True

    Notes
    -----
    The function checks for the existence of the specified CSV file at the beginning and immediately returns
    `False` if the file already exists, ensuring that existing data is not overwritten.
    """

    if dest_csv_path is None:
        dest_csv_path = folder_path

    # Construct the full path for the csv file
    full_csv_path = os.path.join(dest_csv_path, csv_file_name)
    
    # Check if the CSV file already exists
    if os.path.exists(full_csv_path):
        return False

    # Initialize the DataFrame with 'File Path' and 'Complete' columns first
    columns = ['File Path', 'Completed'] + list(default_features.keys())
    df = pd.DataFrame(columns=columns)

    # Identify all .txt or .text files in the folder
    rows_to_add = []
    for file_path in glob.glob(f"{folder_path}/*.txt") + glob.glob(f"{folder_path}/*.text"):
        # Create a new row with default values, setting 'File Path' and 'Complete'
        new_row = {'File Path': file_path, 'Completed': 0}
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