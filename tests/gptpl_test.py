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

def replace_path_if_windows(file_path, old_part, new_part):
    # Check if the current OS is Windows
    # print(os.name)
    if os.name == 'nt':
        # Replace the old part with the new part
        new_file_path = file_path.replace(old_part, new_part)
        return new_file_path
    else:
        # If not Windows, return the original path
        return file_path

books_folder_path = replace_path_if_windows("/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/", "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/", "E:\\NIMBioS\\GPT Parser\\tests\\corpus\\books\\")
openai_key_path = replace_path_if_windows("/Users/natalieharris/UTK/NIMBioS/GPTPipeline/openai_key.txt", "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/", "E:\\NIMBioS\\GPT Parser\\")

with open(openai_key_path, "r") as fd:
    api_key = fd.read()

# setup basic pipeline
pipeline = GPTPipeline(api_key)
# real token length is 16385
pipeline.set_default_values({'delete': False, 'model': 'gpt-3.5-turbo-0125', 'context_window': 16385, 'temperature': 0.0, 'safety multiplier': .95, 'timeout': 15})
generate_primary_csv(books_folder_path, "ebooks.csv", books_folder_path, **{})
pipeline.import_texts(books_folder_path + "ebooks.csv", 1)

# add gpt single prompt module
"""
self.config = gpt_config
        self.input_df = self.config['input df']
        self.output_df = self.config['output df']
        self.prompt = self.config['prompt']
        self.examples = self.config.get('examples', [])
        self.delete = self.config.get('delete', False)
        self.model = self.config.get('model', 'default')
        self.context_window = self.config.get('context window', 'default')
"""
gpt_config = {
    'input df': 'Text List',
    'output df': 'GPT Output',
    'prompt': 'Just respond with \'test successful\' to this request please! Nothing else :)',
    'input text column': 'Full Text'
}
pipeline.add_gpt_singleprompt_module("gpt module", gpt_config)
pipeline.add_df('GPT Output', replace_path_if_windows("/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/gpt_output", "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/", "E:\\NIMBioS\\GPT Parser\\tests\corpus\\books\\"))

pipeline.process("Placeholder Data")

# pipeline.print_modules()
# pipeline.print_dfs()
# pipeline.print_text_df()
pipeline.print_df('GPT Output')