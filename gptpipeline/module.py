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

class GPTModule(Module):
    def __init__(self, gpt_config):
        self.gpt_config = gpt_config

    @abstractmethod
    def process(self, input_data):
        pass

    def make_gpt_request(self, openai_request):
        pass

class GPTSinglePrompt_Module(GPTModule):
    def __init__(self, gpt_config):
        self.gpt_config = gpt_config
    
    def process(self, input_data):
        return "Single module processed: " + input_data

class GPTMultiPrompt_Module(GPTModule):
    def __init__(self, gpt_config):
        self.gpt_config = gpt_config

    def process(self, input_data):
        # GPT specific processing logic
        return "Multi module processed: " + input_data
    
"""
Code Modules can take in zero or more dataframes as input and write to multiple dataframes as output. They can be in any format
"""
class Code_Module(Module):
    def __init__(self, code_config, process_function):
        self.code_config = code_config
        self.process_function = process_function

    def process(self, input_data):
        # Call the provided function with input_data
        processed_data = self.process_function(input_data)
        return processed_data