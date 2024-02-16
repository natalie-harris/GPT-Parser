# gpt_parser/chatgpt_broker.py

import openai
import tiktoken
import time

class ChatGPTBroker:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_tokenized_length(self, system_message, user_message, model, examples=[]):
        """
        Calculate the number of tokens that a text string will be tokenized into 
        by a specific model. Optionally, additional content can be appended to the 
        text from a list of example dictionaries.
        
        Parameters:
        text (str): The input text string to be tokenized.
        model (str): The name or identifier of the model whose tokenizer will be used.
        examples (list of dict, optional): A list of dictionaries where each dictionary 
                                        should have a key "content" with text to append 
                                        to the input text string. Defaults to an empty list.
        
        Returns:
        int: The number of tokens the input text (plus additional content, if provided) 
            is tokenized into by the specified model.
        """
        
        total_text = system_message + user_message

        # Loop through the list of example dictionaries (if provided)
        # and append the content of each example to the input text.
        for example in examples:
            total_text += example["content"]
        
        # Get the encoding (tokenizer) associated with the specified model.
        encoding = tiktoken.encoding_for_model(model)
        
        # Use the encoding (tokenizer) to tokenize the text
        # and then calculate the number of tokens in the tokenized text.
        num_tokens = len(encoding.encode(total_text))
        
        # Return the number of tokens in the tokenized text.
        return num_tokens
    
    # safety multipliers limits max message length just in case tiktoken incorrectly splits tokens
    def split_message_to_lengths(self, system_message, user_message, model, max_context_window, examples=[], safety_multiplier=1.0):
        """
        Returns chunks of text that stay within a specified token limit.
        
        Args:
        - system_message (str): The message to prepend to each chunk of text.
        - user_message (str): The full user message that needs to be split into chunks.
        - model (str): The model being used, as listed on openai's website.
        - max_context_window (int): the maximum number of tokens per chunk
        - examples (list, optional): List of examples for tokenization.
        - safety

        Returns:
        - chunks (lists of strings): A list of chunks, where each chunk is a list containing the system message, a segment of the text, and the end message.
        """

        total_token_length = self.get_tokenized_length(system_message, user_message, model, examples)
        if total_token_length <= max_context_window * safety_multiplier:
            return [system_message, user_message, examples]
        



        
        # else we need to split up the message into chunks. I may have a function that does this in original SBW parser
        return []
    
    #reference for splitting text
    """
    def build_chunk_group(system_message, text, end_message="\n\nEND\n\n", use_gpt4=False, examples=[], just_one_chunk=False, max_context_length=None):
    """"""
    Returns chunks of text that stay within a specified token limit.
    
    Args:
    - system_message (str): The message to prepend to each chunk of text.
    - text (str): The full text that needs to be split into chunks.
    - end_message (str, optional): The message to append to each chunk of text.
    - use_gpt4 (bool, optional): If true, use the token limit for GPT-4.
    - examples (list, optional): List of examples for tokenization.
    - just_one_chunk (bool, optional): If true, return only one chunk.
    - max_context_length (int, optional): If not None, use specified token limit.

    Returns:
    - list: A list of chunks, where each chunk is a list containing the system message, a segment of the text, and the end message.
    """"""

    # Define initial setup values
    system_message_length = len(system_message) + len(end_message)
    max_token_length = 16000  # Default max token length for GPT-3
    if use_gpt4:
        max_token_length = 8000  # GPT-4 token limit
    if max_context_length is not None and max_context_length <= max_token_length:
        max_token_length = max_context_length  # Explicit token limit
    elif max_context_length is not None:
        print(f"Specified maximum context length is too long for GPT. Using {max_token_length} instead.")
    
    base_multiplier = 4
    safety_multiplier = 0.9  # Reduce token size to avoid potential overflows due to local tokenizer differences

    chunk_group = []  # Will hold the resulting chunks of text

    i = 0  # Start index for slicing the text
    while i < len(text):

        # Calculate the length of a user message chunk
        multiplier = base_multiplier
        user_message_length = int(max_token_length * multiplier) - system_message_length

        # Build initial message
        message = system_message + text[i:i+user_message_length] + end_message

        # Assume 'get_tokenized_length' is a function that returns the token count of a message
        token_length = get_tokenized_length(message, 'gpt-3.5-turbo', examples)
        
        # If the token length exceeds the max allowed, reduce the message length and recheck
        while token_length > int(max_token_length * safety_multiplier):
            multiplier *= 0.95
            user_message_length = int(max_token_length * multiplier) - system_message_length
            message = system_message + text[i:i+user_message_length] + end_message
            token_length = get_tokenized_length(message, 'gpt-3.5-turbo', examples)
        
        # Save the chunk and move to the next segment of text
        chunk_group.append([system_message, text[i:i+user_message_length] + end_message])
        i += user_message_length

        # Stop if only one chunk is needed
        if just_one_chunk:
            break

    return chunk_group
    """

    def get_chatgpt_response(self, system_message, user_message, model, model_context_window, temp=0, examples=[], timeout=15):
        """
        Get a response from ChatGPT based on the user and system messages.

        Parameters:
        - system_message (str): The system message to set the behavior of the chat model.
        - user_message (str): The message from the user that the model will respond to.
        - model (str): The GPT model the user wants to use. Models listed at https://platform.openai.com/docs/models.
        - model_context_window (int): Maximum token length for the chosen model. Context windows listed with models at https://platform.openai.com/docs/models.
        - temp (float, optional): Controls the randomness of the model's output (default is 0).
        - examples (list, optional): Additional example messages for training the model (default is an empty list).
        - timeout (int, optional): Controls timeout, in seconds, before the broker stops waiting for a response from OpenAI (default is 15).

        Returns:
        - str: The generated response from the GPT model.
        """

        tokenized_length = self.get_tokenized_length(system_message, user_message, model, examples)
        if tokenized_length > model_context_window:
            return ['Prompt too long...']
        
        # Prepare the messages to send to the Chat API
        new_messages = [{"role": "system", "content": system_message}]
        if len(examples) > 0:
            new_messages.extend(examples)
        new_messages.append({"role": "user", "content": user_message})
        
        # Flag to indicate whether a response has been successfully generated
        got_response = False
        
        # Continue trying until a response is generated
        retries = 0
        max_retries = 10
        while not got_response and retries < max_retries:
            try:
                # Attempt to get a response from the GPT model
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=new_messages,
                    temperature=temp,
                    request_timeout=timeout
                )
                
                # Extract the generated text from the API response
                # generated_text = response['choices'][0]['message']['content']
                generated_text = response
                got_response = True
                return generated_text
                
            except openai.error.RateLimitError as err:
                # Handle rate limit errors
                if 'You exceeded your current quota' in str(err):
                    print("You've exceeded your current billing quota. Go check on that!")
                    return 'BILLING QUOTA ERROR'
                num_seconds = 3
                print(f"Waiting {num_seconds} seconds due to high volume of {model} users.")
                time.sleep(num_seconds)
                
            except openai.error.APIError as err:
                # Handle generic API errors
                print("An error occurred. Retrying request.")
                
            except openai.error.Timeout as err:
                # Handle request timeouts
                num_seconds = 3
                print(f"Request timed out. Waiting {num_seconds} seconds and retrying...")
                retries += 1
                time.sleep(num_seconds)
                
            except openai.error.ServiceUnavailableError as err:
                # Handle service unavailability errors
                num_seconds = 3
                print(f"Server overloaded. Waiting {num_seconds} seconds and retrying request.")
                time.sleep(num_seconds)

        return None