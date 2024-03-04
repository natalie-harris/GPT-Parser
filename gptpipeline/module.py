from abc import ABC, abstractmethod
import pandas as pd
import time
from .helper_functions import get_incomplete_entries, truncate

class Module(ABC):
    def __init__(self, pipeline):
        self.pipeline = pipeline

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
    def __init__(self, pipeline, input_df_name, output_df_name, model=None, context_window=None, safety_multiplier=None, delete=False):
        super().__init__(pipeline)

        #df config
        self.input_df_name = input_df_name
        self.output_df_name = output_df_name

        self.model = model
        self.context_window = context_window
        self.safety_multiplier = safety_multiplier
        self.delete = delete

    @abstractmethod
    def process(self, input_data):
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

    # USE .get() WHEN WE CAN USE DEFAULT VALUES INSTEAD (NOT FOR INPUT/OUTPUT DFS)
    # maybe we should include some sort of move_across operation that moves input_df['selected entry'] to output_df['selected entry']
    # OR we automatically move all across unless there isn't a match or we specify
    # ^ Let's do this for now
    # OR we just add the response to the original df? This wouldn't be bad

    def __init__(self, pipeline, input_df_name, output_df_name, prompt, injection_columns=[], examples=[], model=None, context_window=None, temperature=None, safety_multiplier=None, max_chunks_per_text=None, delete=False, timeout=None, input_text_column='Text', input_completed_column='Completed', output_text_column='Text', output_response_column='Response', output_completed_column='Completed'):
        
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

        # set up output df:
        if self.input_df_name not in pipeline.dfs:
            print("Please instantiate all dfs before instantiating input df")
            exit()

    def setup_df(self, pipeline):

        features_dtypes = pipeline.dfs[self.input_df_name][0].dtypes
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
            pipeline.dfs[self.output_df_name][0][feature] = pd.Series(dtype=object)

        pipeline.dfs[self.output_df_name][0][self.output_text_column] = pd.Series(dtype="string")
        pipeline.dfs[self.output_df_name][0][self.output_response_column] = pd.Series(dtype="string")
        pipeline.dfs[self.output_df_name][0][self.output_completed_column] = pd.Series(dtype="int")

        return True

    def process(self):
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
    def __init__(self, pipeline, code_config, process_function):
        self.pipeline = pipeline
        self.code_config = code_config
        self.process_function = process_function

    def process(self):
        # Call the provided function with input_data
        # process_function needs to return False if it didn't take input from a df, and True if it did
        return self.process_function(self.pipeline)

class Duplication_Module(Module):
    def __init__(self, pipeline, input_df_name, output_df_names):
        self.pipeline = pipeline
        self.input_df_name = input_df_name
        self.output_df_names = output_df_names

        self.input_df = pipeline.get_df(self.input_df_name)
        self.output_dfs = []
        for output_df_name in self.output_df_names:
            self.output_dfs.append(pipeline.get_df(output_df_name))

    def setup_df(self, pipeline):
        features_dtypes = self.input_df[0].dtypes
        features_with_dtypes = list(features_dtypes.items())
        print(features_with_dtypes)

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

        for output_df in self.output_dfs:
            for feature, dtype in zip(features, dtypes):
                output_df[0][feature] = pd.Series(dtype=object)

        pipeline.dfs[self.output_df_name][0][self.output_text_column] = pd.Series(dtype="string")
        pipeline.dfs[self.output_df_name][0][self.output_response_column] = pd.Series(dtype="string")
        pipeline.dfs[self.output_df_name][0][self.output_completed_column] = pd.Series(dtype="int")

        return True
    
    def process(self):
        return False