from abc import ABC, abstractmethod
import pandas as pd
import time

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
Max files that can be in output_df at a time
Number of files read
Number of unprocessed files currently in output_df
"""

class Valve_Module(Module):
    def __init__(self, pipeline, valve_config):
        self.valve_config = valve_config

        self.max_files_total = 1000
        self.max_files_at_once = 1
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

    def process(self, input_data):

        working = False
    
        # print("in valve process!")
        self.current_files = self.output_df[self.output_df['Completed'] == 0]['Source File'].nunique()
        while (self.current_files < self.max_files_at_once and self.total_ran_files < self.max_files_total):

            working = True

            # print(f"{self.current_files} < {self.max_files_at_once};\t\t{self.total_ran_files} < {self.max_files_total}")
            # print("in valve process!")

            # get number of files in processing in text df by checking for unique instances of Source File where Completed = 0
            # print(self.output_df)
            # print(f"Number of files not completed yet: {self.current_files}")

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
            # print(f"Path of entry: {path}")
            with open(path, 'r') as file:
                file_contents = file.read()
            # print(f"File contents: {file_contents}")

            new_entry = [path, file_contents, 0]
            self.output_df.loc[len(self.output_df)] = new_entry
            # self.output_df = pd.concat([self.output_df, new_entry])
            self.total_ran_files += 1

            time.sleep(1)
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