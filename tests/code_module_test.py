from gptpipeline import *
from datetime import datetime

def Module_2_Fnc(input_data):
    now = datetime.now()
    return "Received " + input_data + " at " + now.strftime("%Y-%m-%d %H:%M:%S")

pipeline = GPTPipeline()
pipeline.add_module("Module 2", Code_Module(code_config="some_code_config", process_function=Module_2_Fnc))

result = pipeline.process("Testing code config", 10)
print(result)