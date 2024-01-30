from abc import ABC, abstractmethod

class Module(ABC):
    def __init__(self, module_config):
        self.module_config = module_config

    @abstractmethod
    def process(self, input_data):
        pass

"""
GPT Modules take in a dataframe as input and write to a dataframe as output. 
Two Types of Input Dataframe Format:
1 - Multiple System Prompts: System Prompt | User Prompt | Examples | Complete
2 - Single System Prompt: User Prompt | Complete (System Prompt and Examples are provided elsewhere in module setup, and are applied the same to every user prompt)

NOTE: allow for custom Complete feature name in case multiple modules are accessing the same df
"""
class GPTSinglePrompt_Module(Module):
    def __init__(self, gpt_config):
        self.gpt_config = gpt_config
    
    def process(self, input_data):
        return "Single module processed: " + input_data

class GPTMultiPrompt_Module(Module):
    def __init__(self, gpt_config):
        self.gpt_config = gpt_config

    def process(self, input_data):
        # GPT specific processing logic
        return "Multi module processed: " + input_data
    
"""
Code Modules can take in zero or more dataframes as input and write to a dataframe as output. They can be in any format
"""
class Code_Module(Module):
    def __init__(self, code_config):
        self.code_config = code_config

    def process(self, input_data):
        # Code specific processing logic
        return "Code processed: " + input_data