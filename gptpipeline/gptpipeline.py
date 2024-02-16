from .module import Module, Valve_Module, GPTSinglePrompt_Module, GPTMultiPrompt_Module, Code_Module
from .chatgpt_broker import ChatGPTBroker
from .helper_functions import truncate
from pathlib import Path
import pandas as pd

class GPTPipeline:
    def __init__(self, api_key):
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
        return self.default_vals
    
    def set_default_values(self, default_values):
        for key, value in default_values.items():
            if key in self.default_vals:
                self.default_vals[key] = value

    def get_df(self, name):
        return self.dfs[name][0]

    def add_module(self, name, module):
        if not isinstance(module, Module):
            raise TypeError("Input parameter must be a module")
        self.modules[name] = module

    def add_gpt_singleprompt_module(self, name, config):
        gpt_module = GPTSinglePrompt_Module(pipeline=self, gpt_config=config)
        self.modules[name] = gpt_module

    def add_gpt_multiprompt_module(self, name, config):
        gpt_module = GPTMultiPrompt_Module(pipeline=self, gpt_config=config)

    def add_df(self, name, dest_path, features=['Text', 'Completed']):
        df = pd.DataFrame(columns=features)
        self.dfs[name] = (df, dest_path)

    """
    The text csv contains features: File Name, Completed
    """
    def import_texts(self, path, num_texts):
        files_parent_folder = Path(path).parent.absolute()
        files_df = pd.read_csv(path, sep=',')
        text_df = pd.DataFrame(columns=["Source File", "Full Text", "Completed"])
        self.dfs["Files List"] = (files_df, files_parent_folder)
        self.dfs["Text List"] = (text_df, files_parent_folder)

        # populate Text List with num_texts texts
        # we need to find a way to store one text at a time in memory because we don't wanna use up all our memory
        # if we limit pipeline throughput to one text at a time from text list, we can provide percentage of task complete

        self.add_module("Valve Module", Valve_Module(pipeline=self, valve_config="Valve config"))

        # this implies that we need a new kind of module that reads the texts referenced in files_df and places them in texts_df ONE AT A TIME 
        # (or n-at-a-time, defined by user)
        # reminder that all modules should be running in parallel, and pipeline is considered finished when all modules are idle

    def import_csv(self, name, dest_path): # dest_path must point to the folder that the csv file is located in
        df = pd.read_csv(dest_path + name)
        self.dfs[name] = (df, dest_path)

    # def read_df(self, )

    # We need a maximum for texts to process
    def process(self, input_data, max_texts):
        # Put max_texts (or all texts if total < max_texts) texts into primary df (add completed feature = 0)
        # Use multiple GPT by bridging with code module, or just use single GPT module

        # Set all modules to sequentially process until all of them no longer have any uncompleted processing tasks
        working = True
        while working is True:
            working = False
            for module in self.modules:
                working = self.modules[module].process(input_data)
        print("Finished!")
    
    def print_modules(self):
        print(self.modules)
 
    def print_dfs(self):
        for df in self.dfs:
            print(f"\n{df}:\n {self.dfs[df][0]}")
            # print('')

    def print_df(self, name):
        print(self.dfs[name])

    def print_files_df(self):
        print(self.dfs["Files List"])
    
    def print_text_df(self):
        text_df = self.dfs["Text List"][0]
        for i in range(len(text_df)):
            print(f"Path: {text_df.at[i, 'Source File']}   Full Text: {truncate(text_df.at[i, 'Full Text'], 49)}   Completed: {text_df.at[i, 'Completed']}")

    def process_text(self, system_message, user_message, model='default', model_context_window='default', temp='default', examples=[], timeout='default', safety_multiplier='default'):

        # replace defaults
        if model == 'default':
            model = self.default_vals['model']
        if model_context_window == 'default':
            model_context_window = self.default_vals['context_window']
        if not isinstance(temp, float) or temp > 1.0 or temp < 0.0:
            temp = self.default_vals['temperature']
        if not isinstance(timeout, int) or timeout < 0:
            timeout = self.default_vals['timeout']
        if not isinstance(safety_multiplier, float) or safety_multiplier < 0.0:
            safety_multiplier = self.default_vals['safety multiplier']    

        text_chunks = self.gpt_broker.split_message_to_lengths(system_message, user_message, model, model_context_window, examples, safety_multiplier)
        print(text_chunks)


        return ["Here's some text!"]
    
