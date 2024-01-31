# demonstrates that pandas dfs will be affected by classes that have a reference to that df

import pandas as pd

class DataFrameHolder:
    def __init__(self, df):
        self.df = df

    def modify_dataframe(self):
        # Example modification: adding a new column
        self.df['new_column'] = [1, 2, 3]

# Create a sample DataFrame
original_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# Create an instance of the DataFrameHolder class with the original DataFrame
holder = DataFrameHolder(original_df)

# Modify the DataFrame through the class instance
holder.modify_dataframe()

# Print the original DataFrame after modification
print(original_df)
