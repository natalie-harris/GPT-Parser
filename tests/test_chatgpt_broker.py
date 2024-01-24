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

    @patch('openai.ChatCompletion.create')
    def test_get_chatgpt_response(self, mock_create):
        mock_create.return_value = "Test return"

        response = self.broker.get_chatgpt_response("You are a helpful assistant that responds in beep boop computer language.", "How far away is the sun?", "gpt-3.5-turbo-1106", 16385)
        self.assertEqual(response, "Test return")

        mock_create.assert_called()

if __name__ == '__main__':
    unittest.main()