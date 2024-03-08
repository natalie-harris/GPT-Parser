from setuptools import setup, find_packages

long_description = ''
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name = 'gptpipeline',
    version = '0.0.1',
    author = 'Natalie Harris',
    author_email = 'mzg857@vols.utk.edu',
    description = 'ChatGPT-enabled data processing pipelines',
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
        'openai>=1.9.0,<2.0.0',
        'pandas>=2.2.0,<3.0.0',
        'tqdm>=4.66.0,<5.0.0'
    ],
    python_requires = '>=3.8, <4'
)