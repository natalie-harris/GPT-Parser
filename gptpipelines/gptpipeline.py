from gptpipelines.module import Module, Valve_Module, ChatGPT_Module, Code_Module, Duplication_Module
from gptpipelines.chatgpt_broker import ChatGPTBroker
from gptpipelines.helper_functions import truncate, all_entries_are_true
from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm
import warnings

class GPTPipeline:
    """
    Manages a pipeline for processing data using the ChatGPT API, incorporating various modules
    for specific tasks such as data handling, GPT interactions, and managing data frames.

    Parameters
    ----------
    api_key : str
        The API key required for authenticating requests with the GPT API.
    organization : str, optional
        Organization ID for billing and usage tracking with the OpenAI platform (default is None).
    verbose_chatgpt_api : bool, optional
        If True, enables verbose logging for ChatGPT API interactions (default is False).
    verbose_pipeline_output : bool, optional
        If True, enables verbose logging for pipeline processing output (default is False).

    Attributes
    ----------
    modules : dict
        Maps module names to their respective module instances {Name: module}.
    dfs : dict
        Maps DataFrame names to tuples {Name: (DataFrame, destination path)}, managing input and output data.
    gpt_broker : ChatGPTBroker
        Handles interactions with the ChatGPT API, utilizing the provided API key.
    LOG : logging.Logger
        Configured logger for the pipeline, capturing and formatting log messages.
    default_vals : dict
        Default configuration values for the pipeline, including settings for API interactions.
    """

    def __init__(self, api_key, organization=None, verbose_chatgpt_api=False, verbose_pipeline_output=False):
        """
        Initializes the GPTPipeline with the provided API key.

        This constructor sets up the basic infrastructure required for the pipeline to function,
        including the management of modules, DataFrames, and interactions with the ChatGPT API.

        Parameters
        ----------
        api_key : str
            The API key required for authenticating requests to the GPT API.
        """

        self.modules = {} # {name: module}
        self.dfs = {} # {name: (df, dest_path)}
        self.gpt_broker = ChatGPTBroker(api_key, organization=organization, verbose=verbose_chatgpt_api)

        # Set up basic configuration for logging
        self.LOG = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.default_vals = {
            'delete': False,
            'model': 'No Model Specified', # make sure to check for if no model is specified 
            'context_window': 0,
            'temperature': 0.0,
            'safety multiplier': .95,
            'timeout': 15
        }

        if not verbose_pipeline_output:
            warnings.formatwarning = lambda message, category, filename, lineno, line=None: f'\033[91m{category.__name__}:\033[0m {message}\n'

    def get_default_values(self):
        """
        Get the default pipeline configuration values.

        Returns
        -------
        dict
            The default configuration values.
        """

        return self.default_vals
    
    def set_default_values(self, default_values):
        """
        Set default configuration values.

        Parameters
        ----------
        default_values : dict
            A dictionary of default values to update.
        """

        for key, value in default_values.items():
            if key in self.default_vals:
                self.default_vals[key] = value
            else:
                print(f"'{key}' is not a valid variable name.")

    def add_module(self, name, module):
        """
        Add a module to the pipeline.

        Parameters
        ----------
        name : str
            The name of the module.
        module : Module
            The module instance to add.
        """

        if not isinstance(module, Module):
            raise TypeError("Input parameter must be a module")
        self.modules[name] = module

    def add_chatgpt_module(self, name, input_df_name, output_df_name, prompt, injection_columns=[], examples=[], model=None, context_window=None, temperature=None, safety_multiplier=None, max_chunks_per_text=None, delete=False, timeout=None, input_text_column='Text', input_completed_column='Completed', output_text_column='Text', output_response_column='Response', output_completed_column='Completed'):
        """
        Add a ChatGPT module to the pipeline.

        Parameters
        ----------
        name : str
            The name of the ChatGPT module.
        input_df_name : str
            The name of the input DataFrame.
        output_df_name : str
            The name of the output DataFrame.
        prompt : str
            The prompt to be used by the ChatGPT module.
        injection_columns : list, optional
            Columns from the input DataFrame to inject into the prompt.
        examples : list, optional
            A list of examples to provide context for the GPT model.
        model : str, optional
            The model to use.
        context_window : int, optional
            The context window size for the GPT model.
        temperature : float, optional
            The temperature setting for the GPT model.
        safety_multiplier : float, optional
            The safety multiplier to adjust the maximum token length.
        max_chunks_per_text : int, optional
            The maximum number of chunks into which the input text is split.
        delete : bool, optional
            Whether to delete the input DataFrame after processing.
        timeout : int, optional
            The timeout in seconds for GPT model requests.
        input_text_column : str, optional
            The name of the column containing input text in the input DataFrame.
        input_completed_column : str, optional
            The name of the column indicating whether the input is completed.
        output_text_column : str, optional
            The name of the column for text in the output DataFrame.
        output_response_column : str, optional
            The name of the column for the GPT response in the output DataFrame.
        output_completed_column : str, optional
            The name of the column indicating whether the output is completed.
        """

        gpt_module = ChatGPT_Module(pipeline=self, input_df_name=input_df_name, output_df_name=output_df_name, prompt=prompt, injection_columns=injection_columns, examples=examples, model=model, context_window=context_window, temperature=temperature, safety_multiplier=safety_multiplier, max_chunks_per_text=max_chunks_per_text, delete=delete, timeout=timeout, input_text_column=input_text_column, input_completed_column=input_completed_column, output_text_column=output_text_column, output_response_column=output_response_column, output_completed_column=output_completed_column)
        self.modules[name] = gpt_module

    def add_code_module(self, name, process_function, input_df_names=[], output_df_names=[]):
        """
        Add a code module to the pipeline.

        Parameters
        ----------
        name : str
            The name of the code module.
        process_function : function
            The function to process data within this module.
        input_df_names : list, optional
            If list has df names (str type) in it, their respective dfs will be passed into `input_dfs` list of DataFrames if they are called in user's process_function().
        output_df_names : list, optional
            If list has df names (str type) in it, their respective dfs will be passed into `output_dfs` list of DataFrames if they are called in user's process_function().
        """

        code_module = Code_Module(pipeline=self, process_function=process_function, input_df_names=input_df_names, output_df_names=output_df_names)
        self.modules[name] = code_module

    def add_duplication_module(self, name, input_df_name, output_df_names, input_completed_column='Completed', delete=False):
        """
        Add a duplication module to the pipeline.

        Parameters
        ----------
        name : str
            The name of the duplication module.
        input_df_name : str
            The name of the input DataFrame.
        output_df_names : list
            The names of the output DataFrames.
        input_completed_column : str, optional
            The name of the column indicating whether the input is completed.
        delete : bool, optional
            Whether to delete the input DataFrame after duplication.
        """

        dupe_module = Duplication_Module(pipeline=self, input_df_name=input_df_name, output_df_names=output_df_names, input_completed_column=input_completed_column, delete=delete)
        self.modules[name] = dupe_module

    def add_dfs(self, names, dest_path=None, features={}):
        """
        Add multiple DataFrames to the pipeline.

        Parameters
        ----------
        names : list of str
            The names of the DataFrames to add.
        dest_path : str, optional
            The destination path for the DataFrames. A unique suffix will be added based on the DataFrame name.
        features : dict, optional
            A dictionary specifying the features (columns) and their data types for the new DataFrames.
        """

        for name in names:
            if dest_path is not None:
                new_dest_path = dest_path + "_" + name
                self.add_df(name, dest_path=new_dest_path, features=features)
            else:
                self.add_df(name, features=features)

    def add_df(self, name, dest_path=None, features={}):
        """
        Add a single DataFrame to the pipeline.

        Parameters
        ----------
        name : str
            The name of the DataFrame to add.
        dest_path : str, optional
            The destination path for the DataFrame.
        features : dict, optional
            A dictionary specifying the features (columns) and their data types for the new DataFrame.
        """

        try:
            df = pd.DataFrame(columns=[*features])
            if len(features) != 0:
                df = df.astype(dtype=features)
            self.dfs[name] = (df, dest_path)
        except TypeError:
            print("'Features' format: {'feature_name': dtype, ...}")
            exit()

    def import_texts(self, path, num_texts):
        """
        Import texts from a CSV file and populate DataFrames for file and text lists.

        Parameters
        ----------
        path : str
            The file path to the CSV containing the texts.
        num_texts : int
            The number of texts to import.
        """
        
        files_parent_folder = Path(path).parent.absolute()
        files_df = pd.read_csv(path, sep=',')
        text_df = pd.DataFrame(columns=["Source File", "Full Text", "Completed"])
        self.dfs["Files List"] = (files_df, files_parent_folder)
        self.dfs["Text List"] = (text_df, files_parent_folder)

        self.add_module("Valve Module", Valve_Module(pipeline=self, num_texts=num_texts))

    def import_csv(self, name, dest_path): # dest_path must point to the folder that the csv file is located in
        """
        Import a CSV file into a DataFrame.

        Parameters
        ----------
        name : str
            The name of the DataFrame.
        dest_path : str
            The destination path where the CSV file is located.
        """

        df = pd.read_csv(dest_path + name)
        self.dfs[name] = (df, dest_path)

    def process(self):
        """
        Process all texts through the pipeline, connecting modules to their respective DataFrames and executing processing tasks.
        """

        # Put max_texts (or all texts if total < max_texts) texts into primary df (add completed feature = 0)
        # Use multiple GPT by bridging with code module, or just use single GPT module

        # connect all modules to their respective dfs
        # to be efficient, this requires a network to determine which modules to setup_df first, for now we will just loop until all output dfs are finished setting up
        finished_setup = {}
        for module in self.modules:
            if not isinstance(self.modules[module], Valve_Module):
                finished_setup[module] = False
            else:
                finished_setup[module] = True

        while not all_entries_are_true(finished_setup):
            made_progress = False
            for module in self.modules:
                if isinstance(self.modules[module], Valve_Module) and finished_setup[module] is not True:
                    finished_setup[module] = True
                    made_progress = True
                elif isinstance(self.modules[module], ChatGPT_Module) and finished_setup[module] is not True:
                    result = self.modules[module].setup_df()
                    finished_setup[module] = result
                    made_progress = result or made_progress
                elif isinstance(self.modules[module], Duplication_Module) and finished_setup[module] is not True:
                    result = self.modules[module].setup_df()
                    finished_setup[module] = result
                    made_progress = result or made_progress
                elif isinstance(self.modules[module], Code_Module) and finished_setup[module] is not True:
                    result = self.modules[module].setup_dfs()
                    finished_setup[module] = result
                    made_progress = result or made_progress

            if not made_progress:
                raise RuntimeError("Some dfs cannot be setup")

        # Set all modules to sequentially process until all of them no longer have any uncompleted processing tasks
        working = True
        while working is True:
            working = False
            for module in self.modules:
                working = self.modules[module].process()

        # save each df if dest_path is specified for it

    def print_modules(self):
        """
        Print the list of modules currently added to the pipeline.
        """

        print(self.modules)
 
    def _get_printable_df(self, df_name):
        """
        Retrieves a DataFrame from the pipeline's data store and returns a version suitable for printing.
        
        This method fetches the DataFrame associated with the provided name and processes it to ensure
        that its content is displayed correctly when printed. It replaces newline characters
        in string entries with spaces to avoid disrupting the layout of the printed DataFrame.
        
        Parameters
        ----------
        df_name : str
            The name of the DataFrame to retrieve and process for printing.
        
        Returns
        -------
        pandas.DataFrame
            A copy of the specified DataFrame with newline characters in string entries replaced by spaces.
        
        Notes
        -----
        This method does not modify the original DataFrame stored in the pipeline's data store.
        It operates on and returns a copy of the DataFrame, ensuring that the original data remains unchanged.
        """

        df = self.dfs[df_name][0]
        return df.map(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)

    def print_dfs(self, names=None):
        """
        Print the specified DataFrames. If no names are provided, print all DataFrames.

        Parameters
        ----------
        names : list of str, optional
            The names of the DataFrames to print. If empty, all DataFrames are printed.
        """
        
        if names is None:
            for df_name in self.dfs:
                formatted_df = self._get_printable_df(df_name)
                print(f"\n{df_name}:\n{formatted_df}")

                # print('')
        else:
            for df_name in names:
                formatted_df = self._get_printable_df(df_name)
                print(f"\n{df_name}:\n{formatted_df}")

    def print_df(self, name, include_path=False):
        """
        Print a single DataFrame and optionally its destination path.

        Parameters
        ----------
        name : str
            The name of the DataFrame to print.
        include_path : bool, optional
            Whether to include the destination path in the output.
        """

        formatted_df = self._get_printable_df(name)

        if include_path is False:
            print(formatted_df)
        else:
            print(formatted_df)
            print(self.dfs[name][1])

    # return a df
    def get_df(self, name, include_path=False):
        """
        Retrieve a single DataFrame and optionally its destination path.

        Parameters
        ----------
        name : str
            The name of the DataFrame.
        include_path : bool, optional
            Whether to include the destination path in the return value.

        Returns
        -------
        pd.DataFrame or tuple
            The requested DataFrame, or a tuple containing the DataFrame and its destination path if include_path is True.
        """

        if include_path is False:
            return self.dfs[name][0]
        else:
            return self.dfs[name]

    def print_files_df(self):
        """
        Print the DataFrame containing the list of files.
        """

        print(self.dfs["Files List"])
    
    def print_text_df(self):
        """
        Print the DataFrame containing the list of texts, truncating the full text to a preview length.
        """

        text_df = self.dfs["Text List"][0]
        for i in range(len(text_df)):
            print(f"Path: {text_df.at[i, 'Source File']}   Full Text: {truncate(text_df.at[i, 'Full Text'], 49)}   Completed: {text_df.at[i, 'Completed']}")

    def _prepare_dfs(self, df_names, df_role):
        """
        Prepares a dictionary of DataFrames based on specified names and their intended role.

        This method attempts to fetch each DataFrame by name from the pipeline's `dfs` attribute. 
        If any specified DataFrame is not found, it records the missing DataFrame names. After 
        attempting to gather all specified DataFrames, if any are missing, it issues a warning and returns `None`.

        Parameters
        ----------
        df_names : list of str
            The names of the DataFrames to be prepared. These names should correspond to keys in the pipeline's `dfs` attribute.
        df_role : str
            A descriptive string indicating the role of the specified DataFrames (e.g., 'input' or 'output'). 
            Used for generating meaningful warning messages.

        Returns
        -------
        dict or None
            If all specified DataFrames are found, returns a dictionary where keys are DataFrame names 
            and values are the DataFrame objects. Returns `None` if any specified DataFrames are missing.

        Raises
        ------
        UserWarning
            Warns the user if any of the specified DataFrame names are not found within the pipeline.

        Examples
        --------
        Assuming the pipeline has a DataFrame registered under the name 'sales_data':

        >>> gpt_pipeline._prepare_dfs(['sales_data'], 'input')
        {'sales_data': <DataFrame object>}

        If a specified DataFrame does not exist:

        >>> gpt_pipeline._prepare_dfs(['nonexistent_data'], 'input')
        UserWarning: Specified input DataFrame(s) nonexistent_data not found in pipeline. Please ensure they are created before running process().
        None
        """

        dfs = {}
        missing_dfs = []
        for df_name in df_names:
            try:
                dfs[df_name] = self.dfs[df_name][0]
            except KeyError:
                missing_dfs.append(df_name)
        
        if missing_dfs:
            missing_str = ", ".join(missing_dfs)
            warnings.warn(f"Specified {df_role} DataFrame(s) {missing_str} not found in pipeline. Please ensure they are created before running process().",
                            UserWarning)
            return None
        return dfs

    def process_text(self, system_message, user_message, injections=[], model=None, model_context_window=None, temp=None, examples=[], timeout=None, safety_multiplier=None, max_chunks_per_text=None):
        """
        Process a single text through the GPT broker, handling defaults and injections.

        Parameters
        ----------
        system_message : str
            The system message to send to the GPT model.
        user_message : str
            The user message to process.
        injections : list, optional
            A list of strings to inject into the system message. Useful so that prompts can be somewhat customized for a particular text.
        model : str, optional
            The model to use, None uses the pipeline default.
        model_context_window : int or None, optional
            The context window size, None uses the pipeline default.
        temp : float or None, optional
            The temperature setting for the GPT model, None uses the pipeline default.
        examples : list, optional
            A list of examples to provide context for the GPT model.
        timeout : int or None, optional
            The timeout in seconds for the GPT model request, None uses the pipeline default.
        safety_multiplier : float or None, optional
            The safety multiplier to adjust the maximum token length, None uses the pipeline default.
        max_chunks_per_text : int, optional
            The maximum number of chunks into which the input text is split. Default is all chunks are analyzed.

        Returns
        -------
        list
            A list of tuples containing the processed system message, user message, examples, and GPT response for each chunk.
        """

        # replace defaults
        model = model or self.default_vals['model']
        model_context_window = model_context_window or self.default_vals['context_window']
        if temp is None or not isinstance(temp, float) or temp > 1.0 or temp < 0.0:
            temp = self.default_vals['temperature']
        if timeout is None or not isinstance(timeout, int) or timeout < 0:
            timeout = self.default_vals['timeout']
        if safety_multiplier is None or not isinstance(safety_multiplier, float) or safety_multiplier < 0.0:
            safety_multiplier = self.default_vals['safety multiplier']    

        # inject our injections as a replacement for multiprompt module
        # allows for doing {{}} for edge case when user wants {} in their prompt without injecting into it
        nonplaceholders_count = system_message.count('{{}}')
        placeholders_count = system_message.count('{}')
        placeholders_count = placeholders_count - nonplaceholders_count

        if len(injections) > 0 and len(injections) == placeholders_count:
            system_message = system_message.format(*injections)
        elif len(injections) != placeholders_count:
            print("Inequivalent number of placeholders in system message and injections. Not injecting into system prompt to prevent errors. If you mean to have curly brace sets in your system prompt ({}), then escape them by wrapping them in another set of curly braces ({{}}).")

        # make sure breaking up into chunks is even possible given system message and examples token length
        static_token_length = self.gpt_broker.get_tokenized_length(system_message, "", model, examples)
        if static_token_length >= int(model_context_window * safety_multiplier):
            print(f"The system message and examples are too long for the maximum token length ({int(model_context_window * safety_multiplier)})")
            return ['GPT API call failed.']

        text_chunks = self.gpt_broker.split_message_to_lengths(system_message, user_message, model, model_context_window, examples, safety_multiplier)
        if max_chunks_per_text is not None:
            text_chunks = text_chunks[0:max_chunks_per_text]

        # setup progress bar
        pbar = tqdm(total=len(text_chunks))

        responses = []
        for chunk in text_chunks:
            response = self.gpt_broker.get_chatgpt_response(self.LOG, system_message, chunk, model, model_context_window, temp, examples, timeout)
            responses.append((system_message, chunk, examples, response))
            pbar.update(1)
            pbar.refresh()

        return responses
    
