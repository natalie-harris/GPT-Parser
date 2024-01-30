from .module import Module
import pandas as pd

class GPTPipeline:
    def __init__(self):
        self.modules = {} # {name: module}
        self.dfs = {} # {name: (df, dest_path)}

    def add_module(self, name, module):
        if not isinstance(module, Module):
            raise TypeError("Input parameter must be a module")
        self.modules[name] = module

    def add_df(self, name, dest_path, features):
        new_df = pd.DataFrame(columns=features)
        self.dfs[name] = (new_df, dest_path)

    # We need a maximum for texts to process
    def process(self, input_data, max_texts):
        # Put max_texts (or all texts if total < max_texts) texts into primary df (add completed feature = 0)
        # Use multiple GPT by bridging with code module, or just use single GPT module

        # Set all modules to asyncronously process until all of them no longer have any uncompleted processing tasks

        for module in self.modules:
            input_data = self.modules[module].process(input_data)
        return input_data
    
    def print_modules(self):
        print(self.modules)
 
    def print_dfs(self):
        print(self.dfs)
        # for df in self.dfs:
        #     print('%s > Destination Path: \n' % (df), end='')
        #     # print('')