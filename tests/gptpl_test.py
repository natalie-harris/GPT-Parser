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
    print(os.name)
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

pipeline = GPTPipeline(api_key)
generate_primary_csv(books_folder_path, "ebooks_updated.csv", books_folder_path, **{})
pipeline.import_texts(books_folder_path + "ebooks_updated.csv", 100)
pipeline.process("Placeholder Data", 100)

# pipeline.print_modules()
pipeline.print_text_df()