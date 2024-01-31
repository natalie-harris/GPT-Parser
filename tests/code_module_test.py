from gptpipeline import *
from datetime import datetime

def Module_2_Fnc(input_data):
    now = datetime.now()
    return "Received " + input_data + " at " + now.strftime("%Y-%m-%d %H:%M:%S")

pipeline = GPTPipeline()
pipeline.import_texts("/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/test_csv.csv", 1)
pipeline.add_module("Module 2", Code_Module(pipeline=pipeline, code_config="some_code_config", process_function=Module_2_Fnc))
pipeline.import_csv("test_csv.csv", "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/")
# pipeline.print_dfs()

result = pipeline.process("Testing code config", 10)
print(result)