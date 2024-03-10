"""
Natalie Harris
NIMBioS

This file will help me build the primary functionality of the pipeline.
It will import a bunch of text files containing book excerpts, it will read each excerpt and determine if it contains serious depictions of violence that are inappropriate for general audiences
"""

"""
Notes:
    First df is called "Files List"
    Second df is called "Text List"
"""

from src import *
import pandas as pd
import string

def categorize_books(pipeline, input_dfs, output_dfs):
    working = False

    input_df = input_dfs['GPT Output']
    output_df = pipeline.dfs['Categorized Books'][0]
    incomplete_df = input_df[input_df['Completed'] == 0]
    files = incomplete_df['Source File'].unique().tolist()

    if len(incomplete_df) > 0:
        working = True
        for idx in incomplete_df.index:
            input_df.at[idx, 'Completed'] = 1

    #features={'File Name': object, 'Appropriate': bool}
    for file in files:
        appropriate = True

        filtered_df = input_df[input_df['Source File'] == file]
        responses = filtered_df['Response'].tolist()
        for response in responses:
            response = response.lower().strip()
            response = response.translate(str.maketrans('', '', string.punctuation))
            if len(response) > 4:
                continue
            if response != 'no':
                appropriate = False
        
        next_index = len(output_df)
        output_df.loc[next_index, 'File Name'] = file
        output_df.loc[next_index, 'Appropriate'] = appropriate
        output_df.loc[next_index, 'Completed'] = False


    return working

books_folder_path = "/Users/natalieharris/UTK/NIMBioS/GPTPipelines/tests/corpus/books/"
openai_key_path = "/Users/natalieharris/UTK/NIMBioS/GPTPipelines/openai_key.txt"

with open(openai_key_path, "r") as fd:
    api_key = fd.read()

# setup basic pipeline
pipeline = GPTPipeline(api_key)
pipeline.set_default_values({'delete': False, 'model': 'gpt-3.5-turbo-0125', 'context_window': 16385, 'temperature': 0.0, 'safety multiplier': .95, 'timeout': 15})
generate_primary_csv(books_folder_path, "ebooks.csv", books_folder_path, **{})
pipeline.import_texts(books_folder_path + "ebooks.csv", 1)

# add pipeline modules after valve module
pipeline.add_chatgpt_module("gpt module", input_df_name='Text List', output_df_name='GPT Output', prompt="You are a 'Yes' or 'No' machine. Your sole purpose is to respond with 'Yes' or 'No' to inquiries about whether a text contains serious depictions of violence. It is imperative that your response is strictly limited to 'Yes' or 'No' without any additional commentary or human-like behavior.", input_text_column='Full Text')
pipeline.add_df('GPT Output', dest_path="/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/gpt_output")
pipeline.add_code_module('code module', categorize_books, input_df_names=['GPT Output'])
pipeline.add_df('Categorized Books', dest_path="/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/categorized_books", features={'File Name': object, 'Appropriate': bool, 'Completed': bool})

# run pipeline and print final results
pipeline.process()
pipeline.print_df('GPT Output')
pipeline.print_df('Categorized Books')