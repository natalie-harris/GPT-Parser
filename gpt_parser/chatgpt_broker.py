# gpt_parser/chatgpt_broker.py

import openai
import time

class ChatGPTBroker:
    def __init__(self, api_key):
        self.api_key = api_key

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

        # Combine the system and user messages to evaluate their total tokenized length
        total_message = system_message + user_message
        
        # Select the appropriate GPT model based on the use_gpt4 flag and tokenized length
        # if use_gpt4:
        #     num_tokens = get_tokenized_length(total_message, 'gpt-4', examples)
        #     gpt_model = 'gpt-4'
        # else:
        #     num_tokens = get_tokenized_length(total_message, 'gpt-3.5-turbo', examples)
        #     gpt_model = 'gpt-3.5-turbo' if num_tokens < 4096 else 'gpt-3.5-turbo-16k'

        num_tokens = 10
        
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
