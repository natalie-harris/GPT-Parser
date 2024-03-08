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

books_folder_path = "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/"
openai_key_path = "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/openai_key.txt"

with open(openai_key_path, "r") as fd:
    api_key = fd.read()

# setup basic pipeline
pipeline = GPTPipeline(api_key)
pipeline.set_default_values({'delete': False, 'model': 'gpt-3.5-turbo-0125', 'context_window': 16385, 'temperature': 0.0, 'safety multiplier': .95, 'timeout': 15})
generate_primary_csv(books_folder_path, "ebooks.csv", books_folder_path, **{})
pipeline.import_texts(books_folder_path + "ebooks.csv", 1)

# add pipeline modules after valve module
pipeline.add_duplication_module("dupe module", 'Text List', ['df_1', 'df_2', 'df_3'], delete=True)
pipeline.add_dfs(['df_1', 'df_2', 'df_3'])

# run pipeline and print final results
pipeline.process()
pipeline.print_df("Text List")
pipeline.print_dfs(['df_1', 'df_2', 'df_3'])
