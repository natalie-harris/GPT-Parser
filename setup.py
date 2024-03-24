from setuptools import setup, find_packages

long_description = ''
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name = 'GPTPipelines',
    version = '0.0.1',
    author = 'Natalie Harris',
    author_email = 'mzg857@vols.utk.edu',
    description = 'ChatGPT-enabled data processing pipelines',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    project_urls={
        # 'Documentation': 'https://example.com/GPTPipelines/docs',  # Documentation site is unfinished and not online yet
        'Source Code': 'https://github.com/natalie-harris/GPT-Parser',
        'Bug Tracker': 'https://github.com/natalie-harris/GPT-Parser/issues',
    },
    package_dir={'': 'src'},  # Designate the 'src' directory as where to find packages
    packages=find_packages(where='src'),  # Discover all packages and sub-packages in 'src'
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    install_requires=[
        'openai>=1.9.0,<2.0.0',
        'pandas>=2.2.0,<3.0.0',
        'tqdm>=4.66.0,<5.0.0',
        'tiktoken>=0.6.0,<1.0.0',
        'asciinet>=0.3.1,<0.4.0',
        'networkx>=3.2.1,<4.0.0'
    ],
    python_requires = '>=3.8.0, <4.0.0'
)
