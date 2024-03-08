from src.module import Module, Valve_Module, ChatGPT_Module, Code_Module, Duplication_Module
from src.chatgpt_broker import ChatGPTBroker
from src.helper_functions import truncate, all_entries_are_true
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class GPTPipeline:
    """
    A pipeline for processing tabular data using the ChatGPT API.

    This class initializes a pipeline that can include various modules for data processing,
    handling interactions with the ChatGPT API, and managing DataFrames for input and output data.

    Parameters
    ----------
    api_key : str
        The API key used for authenticating requests to GPT services.

    Attributes
    ----------
    modules : dict
        A dictionary mapping module names to their instances. Modules are components that can be added to the pipeline to perform specific tasks, such as data processing or interaction with the GPT API.
    dfs : dict
        A dictionary mapping DataFrame names to tuples containing the DataFrame itself and an optional destination path for saving the DataFrame. This attribute manages the DataFrames that are used or generated as part of the pipeline's operation.
    gpt_broker : ChatGPTBroker
        An instance of ChatGPTBroker, which handles making requests to the ChatGPT API using the provided API key. It serves as the intermediary for all GPT-related operations within the pipeline.
    default_vals : dict
        A dictionary containing default configuration parameters for the pipeline and its interactions with the GPT API. This includes settings for whether to delete data after processing, the GPT model to use, the context window size, the temperature for generating responses, a safety multiplier to adjust the maximum token length, and a default timeout for API requests.

    Methods
    -------
    __init__(api_key):
        Initializes the GPTPipeline with the specified API key and sets up the initial configuration, including an empty modules dictionary, an empty DataFrame dictionary, a ChatGPTBroker instance, and default configuration values.
    """

    def __init__(self, api_key):
        """
        Initializes the GPTPipeline with the provided API key.

        This constructor sets up the basic infrastructure required for the pipeline to function,
        including the management of modules, DataFrames, and interactions with the GPT API.

        Parameters
        ----------
        api_key : str
            The API key required for authenticating requests to the GPT API.

        """

        self.modules = {} # {name: module}
        self.dfs = {} # {name: (df, dest_path)}
        self.gpt_broker = ChatGPTBroker(api_key)

        self.default_vals = {
            'delete': False,
            'model': 'No Model Specified', # make sure to check for if no model is specified 
            'context_window': 0,
            'temperature': 0.0,
            'safety multiplier': .95,
            'timeout': 15
        }

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

    def add_code_module(self, name, process_function):
        """
        Add a code module to the pipeline.

        Parameters
        ----------
        name : str
            The name of the code module.
        process_function : function
            The function to process data within this module.
        """

        code_module = Code_Module(pipeline=self, code_config="config", process_function=process_function)
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

    """
    The text csv contains features: File Name, Completed
    """
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
                    made_progress = result
                elif isinstance(self.modules[module], Duplication_Module) and finished_setup[module] is not True:
                    result = self.modules[module].setup_df()
                    finished_setup[module] = result
                    made_progress = result
                elif isinstance(self.modules[module], Code_Module) and finished_setup[module] is not True:
                    finished_setup[module] = True
                    made_progress = True

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
 
    def print_dfs(self, names=[]):
        """
        Print the specified DataFrames. If no names are provided, print all DataFrames.

        Parameters
        ----------
        names : list of str, optional
            The names of the DataFrames to print. If empty, all DataFrames are printed.
        """
        
        if len(names) == 0:
            for df in self.dfs:
                print(f"\n{df}:\n {self.dfs[df][0]}")
                # print('')
            return
        
        for df in names:
            print(f"\n{df}:\n {self.dfs[df][0]}")


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

        if include_path is False:
            print(self.dfs[name][0])
        else:
            print(self.dfs[name])

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

    def process_text(self, system_message, user_message, injections=[], model='default', model_context_window='default', temp='default', examples=[], timeout='default', safety_multiplier='default', max_chunks_per_text=None):
        """
        Process a single text through the GPT broker, handling defaults and injections.

        Parameters
        ----------
        system_message : str
            The system message to send to the GPT model.
        user_message : str
            The user message to process.
        injections : list, optional
            A list of strings to inject into the system message.
        model : str, optional
            The model to use, 'default' uses the pipeline default.
        model_context_window : int or 'default', optional
            The context window size, 'default' uses the pipeline default.
        temp : float or 'default', optional
            The temperature setting for the GPT model, 'default' uses the pipeline default.
        examples : list, optional
            A list of examples to provide context for the GPT model.
        timeout : int or 'default', optional
            The timeout in seconds for the GPT model request, 'default' uses the pipeline default.
        safety_multiplier : float or 'default', optional
            The safety multiplier to adjust the maximum token length, 'default' uses the pipeline default.
        max_chunks_per_text : int, optional
            The maximum number of chunks into which the input text is split.

        Returns
        -------
        list
            A list of tuples containing the processed system message, user message, examples, and GPT response for each chunk.
        """

        # replace defaults
        if model is None:
            model = self.default_vals['model']
        if model_context_window is None:
            model_context_window = self.default_vals['context_window']
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
        pbar = tqdm(total=len(text_chunks))  # 100% is the completion

        responses = []
        for chunk in text_chunks:
            response = self.gpt_broker.get_chatgpt_response(system_message, chunk, model, model_context_window, temp, examples, timeout)
            responses.append((system_message, chunk, examples, response))
            pbar.update(1)


        return responses
    
