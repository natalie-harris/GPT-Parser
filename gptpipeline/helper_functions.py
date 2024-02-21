def truncate(string, max_length):
    if max_length > 0 and len(string) > max_length:
        return string[0:max_length-3] + "..."
    else:
        return string
    
def get_incomplete_entries(df, complete_feature):
    """
    Filters the input DataFrame to return rows where the value in the specified column is not 1.
    
    Parameters:
    - df: pandas DataFrame.
    - complete_feature: String specifying the column name to filter by.
    
    Returns:
    - A new DataFrame with rows where the specified column's value is not 1, retaining original indices.
    """
    # Filter the DataFrame based on the condition
    incomplete_df = df[df[complete_feature] != 1]
    
    return incomplete_df

def all_entries_are_true(dictionary):
    for entry in dictionary:
        if dictionary[entry] is False:
            return False
    return True