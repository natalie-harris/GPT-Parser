.. _library_analysis:

Book Analysis Tutorial
======================

This tutorial goes over the basics of setting up a ``GPTPipeline``, performing analysis on a set of books, and saving results to a file.

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

   openai_key_path = "/path/to/your/api/key"


Lastly, we'll define where we want the output data to go. Add the path to where you want the output files saved.

.. code-block:: python

   output_data_path = "/path/to/your/output/data"

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

Analysis in ``GPTPipelines`` works in modules. A module is a piece of code that takes in information from one or more Pandas DataFrames, and spits out information into one or more different DataFrames. A complete ``GPTPipeline`` consists of one or more modules connected in series, with DataFrames in between to facilitate the transfer of information. When you called ``import_texts()``, it automatically created two DataFrames and the aforementioned ``Valve_Module`` in between them. The first DataFrame, called 'Files List' by default, contains all the information from your ``files.csv`` file. By stringing together different modules, you can conduct pretty advanced analysis!

Getting Genres from Texts
.........................

First, we'll attempt to extract the genres of each book. To do that, we need to add a ``ChatGPT_Module`` to the end of the pipeline. A ``ChatGPT_Module`` takes one DataFrame as input and one DataFrame as output. We'll start building this new module here:

.. code-block:: python

   pipeline.add_chatgpt_module(

   )

The first parameter we'll give is the module's name. This is how we reference different modules. Since this module will attempt to extract genres from the texts, we'll just call it ``genre_extractor``. Add this line to your function call:

.. code-block:: python

   name="genre_extractor",

Then, we need to give it the DataFrame the module will use as input. This will be the DataFrame generated by our ``import_texts()`` call that contains the texts from our books. By default, the ``import_texts()`` function names this DataFrame ``Text List``, so that's what we'll call it. Add this line next:

.. code-block:: python

   input_df_name='Text List',

You also need to define where the output will go. Since this DataFrame will contain ChatGPT's output, it makes sense to call it ``GPT Output``:

.. code-block:: python

   output_df_name='GPT Output',

We also need to give the ``ChatGPT_Module`` a prompt that it will respond to. Every time it makes a ChatGPT request, it sends the current text it's analyzing plus the prompt we give it. Creating prompts is an iterative process that can take a while to get just right--This process will be covered in another tutorial. For now, we will use this prompt:

.. code-block:: python

   prompt="This GPT specializes in analyzing excerpts from texts to identify their specific genres, focusing on providing detailed sub-genre classifications. It outputs the three genres, aiming for specificity beyond broad categories, separated by the pipe character (|). This ensures concise and clear responses suitable for parsing by a Python script. The GPT is guided to delve into nuances within the text, seeking out elements that align with specialized niches within known genres, avoiding any extraneous text to facilitate seamless integration with automated processes.",

Lastly, we'll tell the ``ChatGPT_Module`` what column in the input DataFrame the text is located. By default, the ``import_texts()`` function names this column ``Full Text``, so that's how we'll reference it.

.. code-block:: python

   input_text_column='Full Text'


When you're finished, this is what the final ``add_chatgpt_module()`` call should look like:

.. code-block:: python

   pipeline.add_chatgpt_module(
      name="genre_extractor",
      input_df_name='Text List', 
      output_df_name='GPT Output', 
      prompt="This GPT specializes in analyzing excerpts from texts to identify their specific genres, focusing on providing detailed sub-genre classifications. It outputs the three genres, aiming for specificity beyond broad categories, separated by the pipe character (|). This ensures concise and clear responses suitable for parsing by a Python script. The GPT is guided to delve into nuances within the text, seeking out elements that align with specialized niches within known genres, avoiding any extraneous text to facilitate seamless integration with automated processes.", 
      input_text_column='Full Text'
   )

We've successfully added a module! 

Adding a DataFrame
..................

Now, we need to create the module's output DataFrame. Since we named the module's output DataFrame ``GPT Output``, we need to name the DataFrame the same thing. Be sure to add your data destination path too. This tells the DataFrame where it should be saved when analysis is finished:

.. code-block:: python

   pipeline.add_df('GPT Output', dest_folder=output_data_path)

Now, when the ``ChatGPT_Module`` gets a response from ChatGPT, it has a place to put it! 

Formatting the Output
.....................

Technically, the current state of this ``GPTPipeline`` would work. However, ChatGPT has often formats its responses incorrectly.

.. collapse:: book_analysis.py

   .. code-block:: python

      import gptpipelines as gpt

      books_folder_path = "/path/to/your/books/"
      openai_key_path = "/path/to/your/api/key"
      output_data_path = "/path/to/your/output/data"

      # setup basic pipeline
      pipeline = gpt.GPTPipeline(path_to_api_key=openai_key_path)
      pipeline.set_default_values(model='gpt-3.5-turbo', context_window=16385)
      pipeline.import_texts(books_folder_path, "files.csv")

      # add pipeline modules after valve module
      pipeline.add_chatgpt_module(
         name="genre_extractor",
         input_df_name='Text List', 
         output_df_name='GPT Output', 
         prompt="This GPT specializes in analyzing excerpts from texts to identify their specific genres, focusing on providing detailed sub-genre classifications. It outputs the three genres, aiming for specificity beyond broad categories, separated by the pipe character (|). This ensures concise and clear responses suitable for parsing by a Python script. The GPT is guided to delve into nuances within the text, seeking out elements that align with specialized niches within known genres, avoiding any extraneous text to facilitate seamless integration with automated processes.", 
         input_text_column='Full Text'
      )
      pipeline.add_df('GPT Output', dest_folder=output_data_path)


R
-
Using ``GPTPipelines`` in R is currently not supported, but I plan to implement it in the future!
