import csv
import os
import re

# Determine the base path based on the operating system
if os.name == 'nt':  # Windows
    base_path = "E:\\NIMBioS\\GPT Parser\\tests\\corpus\\books\\"
    csv_file_path = base_path + "ebooks.csv"
else:  # macOS and others (assuming macOS for this example)
    base_path = "/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books/"
    csv_file_path = base_path + "ebooks.csv"

# Function to update the file path
def update_file_path(original_path):
    # Regex to extract 'ebook_<n>.txt' part
    ebook_part = re.search(r'ebook_\d+\.txt$', original_path)
    if ebook_part:
        # Construct new path
        return os.path.join(base_path, ebook_part.group())
    else:
        return original_path

# Read the CSV, update paths, and collect rows
updated_rows = []
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Update the 'File Path' for each row
        row['File Path'] = update_file_path(row['File Path'])
        updated_rows.append(row)

# Write the updated rows back to a new CSV to preserve the original for comparison
output_file_path = csv_file_path.replace('ebooks.csv', 'ebooks_updated.csv')
with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = ['File Path', 'Completed']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)

print(f"Updated CSV saved to: {output_file_path}")
