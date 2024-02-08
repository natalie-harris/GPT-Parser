"""
Natalie Harris
NIMBioS

this file will help me build the primary functionality of the pipeline.
It will import a bunch of text files containing book excerpts, it will read each excerpt and determine if it's appropriate to teach the excerpt to a 1st grade classroom
"""

from gptpipeline import *
from datetime import datetime

def Log_Date_and_Time(input_data):
    now = datetime.now()
    return "Received " + input_data + " at " + now.strftime("%Y-%m-%d %H:%M:%S")

with open("/Users/natalieharris/UTK/NIMBioS/GPTPipeline/openai_key.txt", "r") as fd:
    api_key = fd.read()
books_folder_path = "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books"

pipeline = GPTPipeline(api_key)
generate_primary_csv(books_folder_path, "ebooks.csv", books_folder_path, **{})
pipeline.import_texts("/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/ebooks.csv", 100)
pipeline.process("Placeholder Data", 100)

# pipeline.print_modules()
# pipeline.print_dfs()