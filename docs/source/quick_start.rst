.. _quick_start:

Get Started Quickly
===================

This section goes over the basics by showing you how to set up the beginning of a pipeline. If you haven't installed GPTPipelines yet, check out :ref:`installation`.

Importing ``GPTPipelines``
^^^^^^^^^^^^^^^^^^^^^^^^^^

After installing ``GPTPipelines``, we need to import it into our python environment.

.. code-block:: python

   import gptpipelines as gpt

Setup
^^^^^

Next, you'll want to define some paths to your texts, your OpenAI API key (if you plan to use ChatGPT), and your data's destination folder:

.. code-block:: python
    
    text_folder = "/path/to/your/texts"
    openai_key_path = "/path/to/your/api/key" # Optional, only include if you plan to use ChatGPT
    output_data_path = "/path/to/your/output/data"

Then, we'll make a ``GPTPipeline`` object. If you are going to use ChatGPT, you can define the ``path_to_api_key`` parameter in the constructor, and give it your ``openai_key_path`` variable:

.. code-block:: python

    pipeline = gpt.GPTPipeline(path_to_api_key=openai_key_path) # Remove path_to_api_key if you aren't using ChatGPT

Next, you can set up some default values so that you don't have to specify the same information across multiple modules. If you're using ChatGPT, you can specify the ``gpt_model`` and ``gpt_context_window`` parameters that you want to use. You can choose a model and find its context size on `OpenAI's model list <https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo>`__:


.. code-block:: python

    pipeline.set_default_values(gpt_model='gpt-3.5-turbo', gpt_context_window=16385) # Optional, setting gpt_model and gpt_context_window only affects ChatGPT modules

Lastly, you'll set up the beginning of your pipeline by importing your texts! Be sure that all your texts are in the ``text_folder`` you defined, then call ``pipeline.import_texts()``:

.. code-block:: python

    pipeline.import_texts(text_folder)

And that's the beginning of a ``GPTPipeline``! Run this file with ``pipeline.visualize_pipeline()`` at the end to see what your pipeline looks like.

From here, you can add whatever modules you want to achieve your analysis. Be sure to check out :ref:`tutorials` for ideas and guidance!

Here's the completed setup:

.. toggle:: quick_start.py

    .. code-block:: python

        # Import the package
        import gptpipelines as gpt

        # Set some constants
        text_folder = "/path/to/your/texts"
        openai_key_path = "/path/to/your/api/key" # Optional, only include if you plan to use ChatGPT
        output_data_path = "/path/to/your/output/data"

        # Pipeline Setup
        pipeline = gpt.GPTPipeline(path_to_api_key=openai_key_path) # Remove path_to_api_key if you aren't using ChatGPT
        pipeline.set_default_values(gpt_model='gpt-3.5-turbo', gpt_context_window=16385) # Optional, setting gpt_model and gpt_context_window only affects ChatGPT modules
        pipeline.import_texts(text_folder)

        pipeline.visualize_pipeline()        