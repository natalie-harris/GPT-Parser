from gptpipeline import GPTPipeline, generate_primary_csv

generate_primary_csv("./corpus", "test_csv.csv", "./corpus", **{"test_feature": 0, "test_string_feature": "Oh man"})

pipeline = GPTPipeline()

print("Hello Worlb!")