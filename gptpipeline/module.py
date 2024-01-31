from abc import ABC, abstractmethod

class Module(ABC):
    def __init__(self, pipeline, module_config):
        self.module_config = module_config

    @abstractmethod
    def process(self, input_data):
        pass

"""
Valve Module is placed between file df and text df
It limits the amount of texts in text df to n texts, to make sure we don't use up all our memory

Text df automatically deletes texts that are processed (unless specified to save to disk by user)

Internal State:
Max files to read from
Number of files read
"""

class Valve_Module(Module):
    def __init__(self, pipeline, valve_config):
        self.valve_config = valve_config

        self.max_files_total = 10
        self.max_files_at_once = 3
        self.current_files = 0
        self.total_ran_files = 0

        self.input_df = pipeline.get_df("Files List")
        self.output_df = pipeline.get_df("Text List")

        # Make sure we don't try to access files that don't exist
        files_left = self.input_df[0][self.input_df[0]['Completed'] == 0]['File Path'].nunique()
        if (files_left < self.max_files_total):
            print(f"Only {files_left} unprocessed files remaining. Only processing {files_left} for this execution.")
            self.max_files_total = files_left
            if (files_left < self.max_files_at_once):
                self.max_files_at_once = files_left

        print(self.input_df[0])
        print(self.output_df[0])

    def process(self, input_data):
    
        while (self.current_files < self.max_files_at_once and self.total_ran_files < self.max_files_total):

            # get number of files in processing in text df by checking for unique instances of Source File where Completed = 0
            self.current_files = self.output_df[0][self.output_df[0]['Completed'] == 0]['Source File'].nunique()
            print("Number of files not completed yet: " + str(self.current_files))

            # add one file from files list to text list at a time
            # Implement this

        return "Finished processing!"

    

"""
GPT Modules take in a dataframe as input and write to a dataframe as output. 
Two Types of Input Dataframe Format:
1 - Multiple System Prompts: System Prompt | User Prompt | Examples | Complete
2 - Single System Prompt: User Prompt | Complete (System Prompt and Examples are provided elsewhere in module setup, and are applied the same to every user prompt)

NOTE: allow for custom Complete feature name in case multiple modules are accessing the same df
"""

class GPT_Module(Module):
    def __init__(self, pipeline, gpt_config):
        self.gpt_config = gpt_config

    @abstractmethod
    def process(self, input_data):
        pass

    def make_gpt_request(self, openai_request):
        pass

class GPTSinglePrompt_Module(GPT_Module):
    def __init__(self, pipeline, gpt_config):
        self.gpt_config = gpt_config
    
    def process(self, input_data):
        return "Single module processed: " + input_data

class GPTMultiPrompt_Module(GPT_Module):
    def __init__(self, gpt_config):
        self.gpt_config = gpt_config

    def process(self, input_data):
        # GPT specific processing logic
        return "Multi module processed: " + input_data
    
"""
Code Modules can take in zero or more dataframes as input and write to multiple dataframes as output. They can be in any format
"""
class Code_Module(Module):
    def __init__(self, pipeline, code_config, process_function):
        self.code_config = code_config
        self.process_function = process_function

    def process(self, input_data):
        # Call the provided function with input_data
        processed_data = self.process_function(input_data)
        return processed_data