.. _library_analysis:

Book Analysis Tutorial
======================

This tutorial goes over the basics of setting up a ``GPTPipeline``, performing analysis on 

Before you start, go ahead and get an api key from `OpenAI's website <https://platform.openai.com/api-keys>`__. This is necessary because it lets you use ChatGPT for analysis. Copy the key and save it into a file.

Python
------

Installation
^^^^^^^^^^^^

Installing with pip:

.. code-block:: bash

   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ GPTPipelines==0.0.1

Note: The install command is pretty long since the project is currently stored on test.pypi.org instead of pypi.org. We need to add the extra pypi.org index link so that pip can install dependencies like the ``OpenAI`` package. This will change in the future!

Be sure that your ``GPTPipelines`` version is the latest version found on `PyPI <https://test.pypi.org/project/GPTPipelines/0.0.1/#description>`__.


Importing ``GPTPipelines``
^^^^^^^^^^^^^^^^^^^^^^^^^^

After installing ``GPTPipelines``, we need to import it into our python environment.

.. code-block:: python

   import gptpipelines as gpt


Setup
^^^^^

We'll be performing analysis on the top 98 books downloaded from `Project Gutenberg <https://www.gutenberg.org/about/>`__ on March 12th, 2024. Download the books `here <https://drive.google.com/drive/folders/1UMsZpAgY7_c3py-Dpm5hRTupTbsgyv5_?usp=sharing>`__.

Next, define the absolute path to the folder containing the books:

.. code-block:: python

   books_folder_path = "/path/to/your/books/"

and the path to your OpenAI api key:

.. code-block:: python

   openai_key_path = "/path/to/your/api_key"

Then, we'll make a ``GPTPipeline`` object. You can give the ``GPTPipeline`` constructor the plaintext api key, but it's easier and more secure to just pass in the path to the api key:

.. code-block:: python

   pipeline = gpt.GPTPipeline(path_to_api_key=openai_key_path)

After that, we can specify some default values for our pipeline. This is helpful so we don't have to specify the same information for each ChatGPT module we add. For this example, we'll just specify what ChatGPT model to use, plus its context length provided on `OpenAI's model list <https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo>`__. We'll use ``gpt-3.5-turbo``, which strikes a good balance between performance and cost effectiveness, and has a context length of 16,385 tokens in March of 2024.

.. code-block:: python

   pipeline.set_default_values(model='gpt-3.5-turbo', context_window=16385)

.. note::
   
   For `context`, the context length of a model refers to the amount of `tokens <https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them>`__ a model can receive as input. Longer texts can be hundreds of thousands of tokens or more. Since longer texts will often surpass a given model's context length, specifying the context length of the model is necessary so that we know what size chunks the text ought to be broken up into.

Loading Texts
^^^^^^^^^^^^^

Once the pipeline is set up, we can start loading our texts. The ``import_texts()`` function takes at minimum the folder to some text files and a file name. The function compiles each file in the folder you give it and creates a CSV file containing their file paths and whether or not each file has been analyzed. In this case, we'll call our file ``files.csv``, and pass in the path to our directory of books:

.. code-block:: python

   pipeline.import_texts(books_folder_path, "files.csv")

The ``import_texts()`` function also adds a special type of module to the pipeline called a ``Valve_Module``. This is a private class type which accesses the files referenced by the file paths in ``files.csv`` and adds the files' full texts to a new DataFrame.

Now we can get into analysis!


Text Analysis
^^^^^^^^^^^^^

Analysis in ``GPTPipelines`` works in modules. A module is a piece of code that takes in information from one or more Pandas DataFrames, and spits out information into a different DataFrame. After completion, a ``GPTPipeline`` consists of one or more modules connected in series, with DataFrames in between to facilitate the transfer of information. When you called ``import_texts()``, it automatically 


R
-
Using GPTPipelines in R is currently not supported, but I plan to implement it in the future!
