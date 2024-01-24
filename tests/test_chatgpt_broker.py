import sys
from pathlib import Path

# Calculate the path to the directory containing our code
code_path = Path(__file__).parent.parent / 'gpt_parser'
sys.path.append(str(code_path))

import unittest
from unittest.mock import patch
from chatgpt_broker import ChatGPTBroker

class TestChatGPTBroker(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_api_key"
        self.broker = ChatGPTBroker(self.api_key)

    @patch('chatgpt_broker.openai.ChatCompletion.create')
    def test_get_chatgpt_response(self):
        response = self.broker.get_chatgpt_response("", "", "", 1000)
        self.assertEqual(response, "test")

if __name__ == '__main__':
    unittest.main()