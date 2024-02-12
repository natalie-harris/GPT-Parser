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
        total_token_length = self.get_tokenized_length(system_message, user_message, model, examples)
        if total_token_length <= max_context_window * safety_multiplier:
            return [system_message, user_message, examples]
        
        # else we need to split up the message into chunks. I may have a function that does this in original SBW parser
        return []


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
                    exit() # this really shouldn't be here, I need to find a cleaner way to exit
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
