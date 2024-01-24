from setuptools import setup, find_packages

long_description = ''
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name = 'GPT Parser',
    version = '0.0.0',
    author = 'Natalie Harris',
    author_email = 'mzg857@vols.utk.edu',
    description = 'Python library that helps generate a gpt-enabled data processing pipeline',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/natalie-harris/GPT-Parser',
    packages = find_packages(),
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    install_requires=[
        'openai>=1.9.0,<2.0.0'
    ],
    python_requires = '>=3.8, <4'
)