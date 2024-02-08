from gptpipeline import GPTPipeline, generate_primary_csv, GPTSinglePrompt_Module, GPTMultiPrompt_Module, Code_Module

with open("../openai_key.txt", "r") as fd:
    api_key = fd.read()

generate_primary_csv("./corpus", "test_csv.csv", "./corpus", **{"test_feature": 0, "test_string_feature": "Oh man"})
pipeline = GPTPipeline(api_key)
gpt_module = GPTSinglePrompt_Module(gpt_config="some_gpt_config")
code_module = Code_Module(code_config="some_code_config")

pipeline.add_module("Module 1", gpt_module)
pipeline.add_module("Module 2", code_module)

pipeline.add_df('DF 1', '/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus', ['File Name', 'Completed'])

pipeline.print_modules()
pipeline.print_dfs()

# result = pipeline.process("Initial Input")
# print(result)  # This will show the processed input after going through GPTModule and CodeModule