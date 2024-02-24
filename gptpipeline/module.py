from abc import ABC, abstractmethod
import pandas as pd
import time
from .helper_functions import get_incomplete_entries, truncate

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
    def __init__(self, pipeline, num_texts, max_at_once=0):
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

    def process(self, input_data):

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
    def __init__(self, pipeline, gpt_config):
        self.gpt_config = gpt_config

    @abstractmethod
    def process(self, input_data):
        pass

    def make_gpt_request(self, openai_request):
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
class GPTSinglePrompt_Module(GPT_Module):

    # USE .get() WHEN WE CAN USE DEFAULT VALUES INSTEAD (NOT FOR INPUT/OUTPUT DFS)
    # maybe we should include some sort of move_across operation that moves input_df['selected entry'] to output_df['selected entry']
    # OR we automatically move all across unless there isn't a match or we specify
    # ^ Let's do this for now
    # OR we just add the response to the original df? This wouldn't be bad

    def __init__(self, pipeline, gpt_config):
        self.config = gpt_config
        self.pipeline = pipeline
        
        # df config
        self.input_df_name = self.config['input df']
        self.output_df_name = self.config['output df']
        self.delete = self.config.get('delete', 'default')

        self.input_text_column = self.config.get('input text column', 'Text')
        self.input_completed_column = self.config.get('input completed column', 'Completed')
        self.output_text_column = self.config.get('output text column', 'Text')
        self.output_response_column = self.config.get('output response column', 'Response')
        self.output_completed_column = self.config.get('output completed column', 'Completed')

        # important gpt request info
        self.prompt = self.config['prompt']
        self.examples = self.config.get('examples', [])
        
        # remember to replace default vals
        self.model = self.config.get('model', 'default')
        self.context_window = self.config.get('context window', 'default')
        self.temperature = self.config.get('temperature', 'default')
        self.safety_multiplier = self.config.get('safety multiplier', 'default')
        self.timeout = self.config.get('timeout', 'default')

        # set up output df:
        if self.input_df_name not in pipeline.dfs:
            print("Please instantiate all dfs before instantiating input df")
            exit()

    def setup_df(self, pipeline):

        features_dtypes = pipeline.dfs[self.input_df_name][0].dtypes
        features_with_dtypes = list(features_dtypes.items())


        features = []
        dtypes = []

        # Iterate over each item in features_dtypes to separate names and types
        for feature, dtype in features_with_dtypes:
            features.append(feature)
            dtypes.append(dtype)

        if self.input_text_column not in features or self.input_completed_column not in features: # not enough features to setup df with singleprompt module
            # raise ValueError("Input dataframes for Single Prompt GPT modules requires at least 'prompt' and 'completed' features")
            return False

        # move over all features except completed, then add new completed and response column
        # NOTE: WE GOTTA MAKE RESPONSE FEATURE A CUSTOM NAME
        features.remove(self.input_text_column)
        features.remove(self.input_completed_column)

        for feature, dtype in features_with_dtypes:
            pipeline.dfs[self.output_df_name][0][feature] = pd.Series(dtype=object)

        pipeline.dfs[self.output_df_name][0][self.output_text_column] = pd.Series(dtype="string")
        pipeline.dfs[self.output_df_name][0][self.output_response_column] = pd.Series(dtype="string")
        pipeline.dfs[self.output_df_name][0][self.output_completed_column] = pd.Series(dtype="int")

        return True

    def process(self, input_data):
        working = False

        input_df = self.pipeline.get_df(self.input_df_name)
        output_df = self.pipeline.get_df(self.output_df_name)
        incomplete_df = get_incomplete_entries(input_df, self.input_completed_column)

        if len(incomplete_df) > 0:
            entry_index = incomplete_df.index[0]
            entry = input_df.iloc[entry_index]
            text = entry[self.input_text_column]

            print(truncate(text, 49))

            # Put a chatgpt broker call here
            # how does a call have to work?
            # send entire (long) text, break up into chunks, process each system message, user message chunk, examples
            # put each response in its own line in outbreak df, meaning we need to return list of each individual response from gpt broker 
            # then we need to add each entry to output_df

            # ALSO CHECK IF SYSTEM MESSAGE + EXAMPLES >= CONTEXT LENGTH

            responses = self.pipeline.process_text(self.prompt, text, self.model, self.context_window, self.temperature, self.examples, self.timeout, self.safety_multiplier)

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