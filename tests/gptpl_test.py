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
import os

books_folder_path = "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/"
openai_key_path = "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/openai_key.txt"

with open(openai_key_path, "r") as fd:
    api_key = fd.read()

# setup basic pipeline
pipeline = GPTPipeline(api_key)
# real token length is 16385
pipeline.set_default_values({'delete': False, 'model': 'gpt-3.5-turbo-0125', 'context_window': 16385, 'temperature': 0.0, 'safety multiplier': .95, 'timeout': 15})
generate_primary_csv(books_folder_path, "ebooks.csv", books_folder_path, **{})
pipeline.import_texts(books_folder_path + "ebooks.csv", 5)

# add gpt single prompt module

gpt_config = {
    'input df': 'Text List',
    'output df': 'GPT Output',
    'prompt': 'You are a \'Yes\' or \'No\' machine! It is of the utmost importance that your response is ONLY \'Yes\' or \'No\'. Please response with \'Yes\' if this text is appropriate to teach to a 1st grade class, or \'No\' if it isn\'t!',
    'input text column': 'Full Text'
}
pipeline.add_gpt_singleprompt_module("gpt module", gpt_config)
pipeline.add_df('GPT Output', "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/gpt_output")

# pipeline.print_dfs()

pipeline.process("Placeholder Data")

# pipeline.print_modules()
# pipeline.print_dfs()
# pipeline.print_text_df()
# pipeline.print_df('GPT Output')