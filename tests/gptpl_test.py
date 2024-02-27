"""
Natalie Harris
NIMBioS

this file will help me build the primary functionality of the pipeline.
It will import a bunch of text files containing book excerpts, it will read each excerpt and determine if it's appropriate to teach the excerpt to a 1st grade classroom
"""

"""
Notes:
    First df is called "Files List"
    Second df is called "Text List"
"""

from gptpipeline import *
import pandas as pd
import string
import os

def categorize_books(pipeline):
    working = False

    input_df = pipeline.get_df('GPT Output')
    output_df = pipeline.get_df('Categorized Books')
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
            print(response)
            if response != 'no':
                appropriate = False
        
        next_index = len(output_df)
        output_df.loc[next_index, 'File Name'] = file
        output_df.loc[next_index, 'Appropriate'] = appropriate
        output_df.loc[next_index, 'Completed'] = False


    return working

books_folder_path = "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/"
openai_key_path = "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/openai_key.txt"

with open(openai_key_path, "r") as fd:
    api_key = fd.read()

# setup basic pipeline
pipeline = GPTPipeline(api_key)
# real token length is 16385
pipeline.set_default_values({'delete': False, 'model': 'gpt-3.5-turbo-0125', 'context_window': 16385, 'temperature': 0.0, 'safety multiplier': .95, 'timeout': 15})
generate_primary_csv(books_folder_path, "ebooks.csv", books_folder_path, **{})
pipeline.import_texts(books_folder_path + "ebooks.csv", 25)

# add gpt single prompt module
gpt_config = {
    'input df': 'Text List',
    'output df': 'GPT Output',
    'prompt': "You are a 'Yes' or 'No' machine. It is of the utmost importance that your response is ONLY \'Yes\' or \'No\'. Please response with 'Yes' if this text has serious depictions of violence in it, or 'No' if it doesn't. I know you are a model, do not act like a human by adding anything else. 'Yes' or 'No' only.",
    'input text column': 'Full Text'
}
pipeline.add_gpt_singleprompt_module("gpt module", gpt_config)
pipeline.add_df('GPT Output', "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/gpt_output")
pipeline.add_code_module('code module', categorize_books)
pipeline.add_df('Categorized Books', dest_path="/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/categorized_books", features={'File Name': object, 'Appropriate': bool, 'Completed': bool})

pipeline.process()

pipeline.print_df('Categorized Books')