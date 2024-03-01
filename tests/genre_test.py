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

from gptpipeline import *
import pandas as pd
import string

books_folder_path = r"C:\\Users\\natal\\OneDrive\Documents\\GitHub\\GPTPipeline\\tests\\corpus\books\\"
openai_key_path = r"C:\Users\natal\OneDrive\Documents\GitHub\GPTPipeline\openai_key.txt"

with open(openai_key_path, "r") as fd:
    api_key = fd.read()

# setup basic pipeline
pipeline = GPTPipeline(api_key)
pipeline.set_default_values({'delete': False, 'model': 'gpt-3.5-turbo-0125', 'context_window': 16385, 'temperature': 0.0, 'safety multiplier': .95, 'timeout': 15})
generate_primary_csv(books_folder_path, "ebooks_updated.csv", books_folder_path, **{})
pipeline.import_texts(books_folder_path + "ebooks_updated.csv", 3)

# add pipeline modules after valve module
pipeline.add_chatgpt_module("genre finder", input_df_name='Text List', output_df_name='Genre Output', prompt="This GPT specializes in analyzing excerpts from books to identify their specific genres, focusing on providing detailed sub-genre classifications. It outputs the three genres, aiming for specificity beyond broad categories, separated by the pipe character (|). This ensures concise and clear responses suitable for parsing by a Python script. The GPT is guided to delve into nuances within the text, seeking out elements that align with specialized niches within known genres, avoiding any extraneous text to facilitate seamless integration with automated processes.", input_text_column='Full Text', max_chunks_per_text=1)
pipeline.add_df('Genre Output')
pipeline.add_chatgpt_module("genre formatter", input_df_name='Genre Output', output_df_name='Formatted Genre Output', input_text_column='Response', output_text_column='Genres', output_response_column='Formatted Genres', prompt="Genre Formatter is exclusively focused on formatting input text into a standardized genre listing form, outputting as 'genre 1 | genre 2 | genre 3'. It strictly adheres to this format, excluding any additional natural language, interpretations, or clarifications. The emphasis is on precision, ensuring the output strictly follows the specified format without ambiguity. Angle brackets, used in the description to indicate placeholders, are not included in the actual output. If the input doesn't clearly match the expected format, Genre Formatter will not respond, maintaining a straightforward approach by only producing the formatted genres without any additional communication or error messages.")
pipeline.add_df('Formatted Genre Output')

# run pipeline and print final results
pipeline.process()
pipeline.print_df('Formatted Genre Output')
