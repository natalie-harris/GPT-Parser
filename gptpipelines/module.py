from abc import ABC, abstractmethod
import pandas as pd
import time
from gptpipelines.helper_functions import get_incomplete_entries, truncate
import inspect
import warnings

class Module(ABC):
    """
    An abstract base class for a pipeline module.
    
    This class defines the structure for modules that can be added to a GPTPipeline
    for processing data.

    Attributes
    ----------
    pipeline : GPTPipeline
        Reference to the GPTPipeline instance that the module is part of.
    """

    def __init__(self, pipeline):
        """
        Initialize a Module instance.

        Parameters
        ----------
        pipeline : GPTPipeline
            The pipeline instance to which the module belongs.
        """

        self.pipeline = pipeline

    @abstractmethod
    def process(self):
        """
        Abstract method to process input data through the module.

        This method must be implemented by subclasses.
        """

        pass

"""
Valve Module is placed between file df and text df
It limits the amount of texts in text df to n texts, to make sure we don't use up all our memory

Text df automatically deletes texts that are processed (unless specified to save to disk by user)

Internal State:
Max files to read from
Max files that can be in output_df at a time
Number of files read
Number of unprocessed files currently in output_df
"""

class Valve_Module(Module):
    """
    A module to limit the number of texts processed to prevent memory overflow.

    This module is placed between the file DataFrame and text DataFrame and manages
    the flow of texts to ensure that the pipeline does not exceed memory limitations.

    Attributes
    ----------
    Inherits all attributes from the Module class.

    max_files_total : int
        The maximum number of files to read from the input.
    max_files_at_once : int
        The maximum number of files that can be in the output DataFrame at a time.
    current_files : int
        The current number of unprocessed files in the output DataFrame.
    total_ran_files : int
        The total number of files processed so far.
    input_df : pd.DataFrame
        The input DataFrame containing file information.
    output_df : pd.DataFrame
        The output DataFrame where texts are stored.
    """

    def __init__(self, pipeline, num_texts, max_at_once=0):
        """
        Initializes a Valve_Module instance.

        Parameters
        ----------
        pipeline : GPTPipeline
            The pipeline instance to which the module belongs.
        num_texts : int
            The maximum number of texts to process in total.
        max_at_once : int, optional
            The maximum number of texts to hold in memory at once. Defaults to 0, which is treated as no limit.
        """

        super().__init__(pipeline)

        self.max_files_total = num_texts
        if max_at_once >= 1:
            self.max_files_at_once = max_at_once
        else:
            self.max_files_at_once = self.max_files_total
        self.current_files = 0
        self.total_ran_files = 0

        self.input_df = pipeline.get_df("Files List")
        self.output_df = pipeline.get_df("Text List")

        # Make sure we don't try to access files that don't exist
        files_left = self.input_df[self.input_df['Completed'] == 0]['File Path'].nunique()
        if files_left == 0:
            print("There are no files left to be processed.")
        elif (files_left < self.max_files_total):
            file_plural = "file" if files_left == 1 else "files"
            print(f"Only {files_left} unprocessed {file_plural} remaining. Only processing {files_left} {file_plural} on this execution.")
            self.max_files_total = files_left
            if (files_left < self.max_files_at_once):
                self.max_files_at_once = files_left

        # print(self.input_df)
        # print(self.output_df)

    def process(self):
        """
        Processes input data to limit the number of texts in the output DataFrame.

        Overrides the abstract process method in the Module class.

        Returns
        -------
        bool
            True if processing occurred, indicating that there were texts to process; False otherwise.
        """

        working = False

        # get number of files in processing in text df by checking for unique instances of Source File where Completed = 0
        self.current_files = self.output_df[self.output_df['Completed'] == 0]['Source File'].nunique()
        while (self.current_files < self.max_files_at_once and self.total_ran_files < self.max_files_total):

            working = True

            # add one file from files list to text list at a time
            has_unprocessed_files = (self.input_df['Completed'] == False).any()
            if not has_unprocessed_files:
                break

            # Find the index of the first entry where 'Completed' is False
            row_index = self.input_df[self.input_df['Completed'] == False].index[0]
            # Set the 'Completed' feature of that entry to True
            self.input_df.at[row_index, 'Completed'] = 1

            # Get the text at the file referenced in File Path
            entry = self.input_df.loc[row_index]
            path = entry['File Path']
            with open(path, 'r', encoding='utf-8') as file:
                file_contents = file.read()

            new_entry = [path, file_contents, 0]
            self.output_df.loc[len(self.output_df)] = new_entry
            # self.output_df = pd.concat([self.output_df, new_entry])
            self.total_ran_files += 1

            # time.sleep(1)
            self.current_files = self.output_df[self.output_df['Completed'] == 0]['Source File'].nunique()

            # print(f"Output df: [[[\n{self.output_df}\n]]]")

        # print(f"{self.current_files} < {self.max_files_at_once};\t\t{self.total_ran_files} < {self.max_files_total}")

        return working

"""
GPT Modules take in a dataframe as input and write to a dataframe as output. 
Two Types of Input Dataframe Format:
1 - Multiple System Prompts: System Prompt | User Prompt | Examples | Complete
2 - Single System Prompt: User Prompt | Complete (System Prompt and Examples are provided elsewhere in module setup, and are applied the same to every user prompt)

NOTE: allow for custom Complete feature name in case multiple modules are accessing the same df
"""

class GPT_Module(Module):
    """
    An abstract base class for GPT modules.

    This class extends Module to define a structure for modules that interact with
    GPT models for processing text data.

    Attributes
    ----------
    Inherits all attributes from the Module class.

    input_df_name : str
        The name of the input DataFrame.
    output_df_name : str
        The name of the output DataFrame.
    model : str, optional
        The GPT model to use.
    context_window : int, optional
        The context window size for the GPT model.
    safety_multiplier : float, optional
        A multiplier to adjust the maximum token length for safety.
    delete : bool
        Whether to delete entries from the input DataFrame after processing.
    """

    def __init__(self, pipeline, input_df_name, output_df_name, model=None, context_window=None, safety_multiplier=None, delete=False):
        """
        Initializes a GPT_Module instance.

        Parameters
        ----------
        pipeline : GPTPipeline
            The pipeline instance to which the module belongs.
        input_df_name : str
            The name of the input DataFrame.
        output_df_name : str
            The name of the output DataFrame.
        model : str, optional
            The GPT model to use. Default is None.
        context_window : int, optional
            The context window size for the GPT model. Default is None.
        safety_multiplier : float, optional
            A multiplier to adjust the maximum token length for safety. Default is None.
        delete : bool, optional
            Whether to delete entries from the input DataFrame after processing. Default is False.
        """

        super().__init__(pipeline)

        #df config
        self.input_df_name = input_df_name
        self.output_df_name = output_df_name

        self.model = model
        self.context_window = context_window
        self.safety_multiplier = safety_multiplier
        self.delete = delete

    @abstractmethod
    def process(self):
        """
        Abstract method to process input data through the GPT model.

        This method must be implemented by subclasses.
        """

        pass

"""
gpt_config: dictionary: {
        input_df (str)
        output_df (str)
        delete (bool)
        model (str)
        context_window (int)
        temp (float)
        prompt (str)
        examples (dict) # not implemented yet
    }
"""
class ChatGPT_Module(GPT_Module):
    """
    A module designed to process texts through a ChatGPT model.

    This module takes input data from a specified DataFrame, processes it through a ChatGPT model,
    and outputs the results to another DataFrame.

    Attributes
    ----------
    Inherits all attributes from the GPT_Module class.

    prompt : str
        The GPT prompt to be used for all entries.
    injection_columns : list of str
        Columns from the input DataFrame whose values are injected into the prompt.
    examples : list
        A list of examples provided to the GPT model for context.
    temperature : float, optional
        The temperature setting for the GPT model. Default is None.
    max_chunks_per_text : int, optional
        The maximum number of chunks into which the input text is split. Default is None.
    timeout : int, optional
        The timeout in seconds for GPT model requests. Default is None.
    input_text_column : str
        The name of the column in the input DataFrame containing the text to be processed.
    input_completed_column : str
        The name of the column in the input DataFrame that marks whether the entry has been processed.
    output_text_column : str
        The name of the column in the output DataFrame for storing text.
    output_response_column : str
        The name of the column in the output DataFrame for storing the GPT model's response.
    output_completed_column : str
        The name of the column in the output DataFrame that marks whether the entry has been processed.
    """

    def __init__(self, pipeline, input_df_name, output_df_name, prompt, injection_columns=[], examples=[], model=None, context_window=None, temperature=None, safety_multiplier=None, max_chunks_per_text=None, delete=False, timeout=None, input_text_column='Text', input_completed_column='Completed', output_text_column='Text', output_response_column='Response', output_completed_column='Completed'):
        """
        Initializes a ChatGPT_Module instance with specified configuration.

        Parameters
        ----------
        Inherits all parameters from the GPT_Module class and introduces additional parameters for ChatGPT module configuration.
        """
        
        super().__init__(pipeline=pipeline,input_df_name=input_df_name,output_df_name=output_df_name, model=model, context_window=context_window,safety_multiplier=safety_multiplier,delete=False)

        self.max_chunks_per_text = max_chunks_per_text
        self.temperature=temperature
        self.timeout=timeout
        
        self.input_text_column = input_text_column
        self.input_completed_column = input_completed_column
        self.output_text_column = output_text_column
        self.output_response_column = output_response_column
        self.output_completed_column = output_completed_column

        # important gpt request info
        self.prompt = prompt
        self.examples = examples
        self.injection_columns = injection_columns

    def setup_df(self):
        """
        Sets up the input and output DataFrames based on module configuration.

        Returns
        -------
        bool
            True if setup is successful, False otherwise.
        """

        self.input_df = self.pipeline.get_df(self.input_df_name)
        self.output_df = self.pipeline.get_df(self.output_df_name)

        if self.input_text_column not in self.input_df.columns or self.input_completed_column not in self.input_df.columns:
            return False

        features_dtypes = self.pipeline.dfs[self.input_df_name][0].dtypes
        features_with_dtypes = list(features_dtypes.items())

        # print(f"FEATURES: {features_with_dtypes}")
        # print(f"{self.input_text_column}")
        # print(f"{self.input_completed_column}")

        features = []
        dtypes = []

        # Iterate over each item in features_dtypes to separate names and types
        for feature, dtype in features_with_dtypes:
            if feature != self.input_completed_column and feature != self.input_text_column:
                features.append(feature)
                dtypes.append(dtype)

        for feature, dtype in zip(features, dtypes):
            self.pipeline.dfs[self.output_df_name][0][feature] = pd.Series(dtype=object)

        self.pipeline.dfs[self.output_df_name][0][self.output_text_column] = pd.Series(dtype="string")
        self.pipeline.dfs[self.output_df_name][0][self.output_response_column] = pd.Series(dtype="string")
        self.pipeline.dfs[self.output_df_name][0][self.output_completed_column] = pd.Series(dtype="int")

        return True

    def process(self):
        """
        Processes the input DataFrame through the ChatGPT model based on the module configuration.

        Overrides the abstract process method in the GPT_Module class.

        Returns
        -------
        bool
            True if processing occurred, indicating that there were texts to process; False otherwise.
        """

        working = False

        input_df = self.pipeline.get_df(self.input_df_name)
        output_df = self.pipeline.get_df(self.output_df_name)
        incomplete_df = get_incomplete_entries(input_df, self.input_completed_column)

        if len(incomplete_df) > 0:
            entry_index = incomplete_df.index[0]
            entry = input_df.iloc[entry_index]
            text = entry[self.input_text_column]

            injections = []
            for column in self.injection_columns:
                injections.append(entry[column])

            print(truncate(text, 49))

            # Put a chatgpt broker call here
            # how does a call have to work?
            # send entire (long) text, break up into chunks, process each system message, user message chunk, examples
            # put each response in its own line in outbreak df, meaning we need to return list of each individual response from gpt broker 
            # then we need to add each entry to output_df

            # ALSO CHECK IF SYSTEM MESSAGE + EXAMPLES >= CONTEXT LENGTH

            responses = self.pipeline.process_text(self.prompt, text, injections, self.model, self.context_window, self.temperature, self.examples, self.timeout, self.safety_multiplier, self.max_chunks_per_text)

            # we don't need to include system message or examples for singleprompt module since they are static
            for system_message, chunk, examples, response in responses:
                # Assuming 'entry' is a Series, convert it to a one-row DataFrame
                new_entry_df = entry.to_frame().transpose().copy()
                
                # Drop the unnecessary columns
                new_entry_df = new_entry_df.drop(columns=[self.input_text_column, self.input_completed_column])
                
                # Add the new data
                new_entry_df[self.output_text_column] = chunk
                new_entry_df[self.output_response_column] = response
                new_entry_df[self.output_completed_column] = 0
                
                # Identify the next index for output_df
                next_index = len(output_df)
                
                # Iterate over columns in new_entry_df to add them to output_df
                for col in new_entry_df.columns:
                    output_df.at[next_index, col] = new_entry_df[col].values[0]

            if len(responses) != 0:
                input_df.at[entry_index, self.input_completed_column] = 1
                working = True

        return working
   
"""
Code Modules can take in zero or more dataframes as input and write to multiple dataframes as output. They can be in any format
"""
class Code_Module(Module):
    """
    A module for executing custom code as part of the pipeline.

    This module allows for the execution of arbitrary Python functions, facilitating
    custom data processing or transformation within the pipeline.

    Attributes
    ----------
    Inherits all attributes from the Module class.

    code_config : various
        Configuration data or parameters for the custom code execution.
    process_function : function
        The custom function to be executed by the module.
    """

    def __init__(self, pipeline, process_function, input_df_names=[], output_df_names=[]):
        """
        Initializes a Code_Module instance with specified custom code and configuration.

        Parameters
        ----------
        pipeline : GPTPipeline
            The pipeline instance to which the module belongs.
        code_config : various
            Configuration data or parameters for the custom code execution.
        process_function : function
            The custom function to be executed by the module.
        """

        super().__init__(pipeline=pipeline)
        self.process_function = process_function
        self.input_df_names = input_df_names
        self.output_df_names = output_df_names
        self.func_args = {}

    def setup_dfs(self):
        """
        Prepares and validates the input and output DataFrames for the code module.

        This method inspects the parameters of the user-defined process function to determine if `input_dfs` 
        and/or `output_dfs` are expected. It then attempts to prepare these DataFrame dictionaries based on the 
        DataFrame names provided at module initialization. If any specified DataFrames are missing, it issues a warning 
        and returns False to indicate unsuccessful setup.

        Returns
        -------
        bool
            Returns True if all specified DataFrames are successfully prepared and assigned. 
            Returns False if any specified DataFrames are missing, indicating that the setup was unsuccessful.

        Notes
        -----
        - The method utilizes a pipeline-level function `_prepare_dfs` to gather the DataFrames by their names. 
        This function should return a dictionary of DataFrames if all specified names are found, or `None` if any 
        are missing.
        - Warnings are issued if the user's function expects `input_dfs` or `output_dfs` but the respective DataFrame 
        names were not specified at module addition, or if the specified DataFrame names do not exist in the pipeline.

        Examples
        --------
        >>> # Assuming 'input_df_names' were specified as ['sales_data'] during module initialization
        >>> # and 'sales_data' DataFrame exists in the pipeline
        >>> module.setup_df()
        True

        >>> # Assuming 'input_df_names' were specified as ['missing_data'] during module initialization,
        >>> # but 'missing_data' DataFrame does not exist in the pipeline
        >>> module.setup_df()
        UserWarning: Specified input DataFrame(s) 'missing_data' not found in pipeline. Please ensure they are created before running process().
        False
        """

        # Inspect the process_function parameters right in the __init__ method
        params = inspect.signature(self.process_function).parameters

        # Check for 'input_dfs' parameter and prepare if specified
        if 'input_dfs' in params:
            if not self.input_df_names:
                warnings.warn("You've requested 'input_dfs' in your function but did not specify any input_df_names when adding the code module.",
                            UserWarning)
                self.func_args['input_dfs'] = {}
            else:
                input_dfs = self.pipeline._prepare_dfs(self.input_df_names, 'input')
                if input_dfs is None:  # Missing one or more specified input_dfs
                    return False
                self.func_args['input_dfs'] = input_dfs

        # Check for 'output_dfs' parameter and prepare if specified
        if 'output_dfs' in params:
            if not self.output_df_names:
                warnings.warn("You've requested 'output_dfs' in your function but did not specify any output_df_names when adding the code module.",
                            UserWarning)
                self.func_args['output_dfs'] = {}
            else:
                output_dfs = self.pipeline._prepare_dfs(self.output_df_names, 'output')
                if output_dfs is None:  # Missing one or more specified output_dfs
                    return False
                self.func_args['output_dfs'] = output_dfs

        return True

    def process(self):
        """
        Executes the custom code function provided during initialization.

        Overrides the abstract process method in the Module class.

        Returns
        -------
        bool
            The return value of the custom process_function, typically True if processing occurred and False otherwise.
        """

        # NOTE: In user guides, be sure to explain that in order to receive an easy-to-access list of requested dfs, 
        # the user function has to include either/both 'input_dfs' or 'output_dfs' EXACTLY in the the parameters
        # list, and include a proper list of its associated df names in add_code_module.

        # Dynamically call the user's process function with the prepared arguments
        result = self.process_function(self.pipeline, **self.func_args)
        return result
    
class Duplication_Module(Module):
    """
    A module for duplicating entries from an input DataFrame to multiple output DataFrames.

    This module is designed to facilitate the copying of data across different parts of the pipeline, 
    ensuring that data can be processed in parallel or stored for different purposes without altering the original source.

    Attributes
    ----------
    Inherits all attributes from the Module class.

    input_df_name : str
        The name of the input DataFrame from which entries will be duplicated.
    output_df_names : list of str
        The names of the output DataFrames to which entries will be duplicated.
    input_completed_column : str
        The name of the column in the input DataFrame that marks whether the entry has been processed.
    delete : bool
        Whether to delete entries from the input DataFrame after duplication.
    """

    def __init__(self, pipeline, input_df_name, output_df_names, input_completed_column='Completed', delete=False):
        """
        Initializes a Duplication_Module instance with specified configuration for data duplication.

        Parameters
        ----------
        pipeline : GPTPipeline
            The pipeline instance to which the module belongs.
        input_df_name : str
            The name of the input DataFrame from which entries will be duplicated.
        output_df_names : list of str
            The names of the output DataFrames to which entries will be duplicated.
        input_completed_column : str
            The name of the column in the input DataFrame that marks whether the entry has been processed.
        delete : bool
            Whether to delete entries from the input DataFrame after duplication.
        """

        super().__init__(pipeline=pipeline)
        self.input_df_name = input_df_name
        self.output_df_names = output_df_names
        self.input_completed_column=input_completed_column
        self.delete = delete

    def setup_df(self):
        """
        Prepares the input and output DataFrames for the duplication process.

        Returns
        -------
        bool
            True if setup is successful, False otherwise.
        """

        self.input_df = self.pipeline.get_df(self.input_df_name)
        self.output_dfs = []
        for output_df_name in self.output_df_names:
            self.output_dfs.append(self.pipeline.get_df(output_df_name))
        
        num_features = self.input_df.shape[1]
        if num_features <= 1: # The input df is empty or just has a completed column, so we can't duplicate
            return False
        elif self.input_completed_column not in self.input_df.columns: # Make sure that the completed column is here 
            return False

        features_dtypes = self.input_df.dtypes
        features_with_dtypes = list(features_dtypes.items())
        print(features_with_dtypes)

        # print(f"FEATURES: {features_with_dtypes}")
        # print(f"{self.input_text_column}")
        # print(f"{self.input_completed_column}")

        features = []
        dtypes = []

         # Iterate over each item in features_dtypes to separate names and types
        for feature, dtype in features_with_dtypes:
            features.append(feature)
            dtypes.append(dtype)

        for output_df in self.output_dfs:
            for feature, dtype in zip(features, dtypes):
                output_df[feature] = pd.Series(dtype=dtype)

        return True
    
    def process(self):
        """
        Duplicates entries from the input DataFrame to each specified output DataFrame.

        Overrides the abstract process method in the Module class.

        Returns
        -------
        bool
            True if duplication occurred, indicating that there were entries to duplicate; False otherwise.
        """

        working = False

        incomplete_entries = get_incomplete_entries(self.input_df, self.input_completed_column)
        while (len(incomplete_entries) > 0): # while there are any incomplete entries in the input df
            row_index = incomplete_entries.index[0]
            entry = self.input_df.iloc[row_index].values.tolist()
            for output_df in self.output_dfs:
                output_df.loc[len(output_df)] = entry

            if not self.delete:
                self.input_df.at[row_index, 'Completed'] = 1
            else:
                self.input_df.drop(row_index, inplace=True)

            working = True

            incomplete_entries = get_incomplete_entries(self.input_df, self.input_completed_column)

        return working