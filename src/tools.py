from langchain_community.tools import Tool
from typing import Dict, Any
import json


class TravelTools:
    def __init__(self, data_path: str):
        """
        Initialize the TravelTools class.
        Args:
            data_path (str): Path to the JSON file containing all data.
        """
        self.data_path = data_path
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        """
        Load all data from the specified JSON file.
        Returns:
            Dict[str, Any]: The parsed JSON data.
        """
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}.")
            return {}
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON file.")
            return {}

    def get_data_by_key(self, key: str) -> Any:
        """
        Retrieve data for a specific key from the JSON file.
        Args:
            key (str): The key to retrieve from the data.
        Returns:
            Any: The value corresponding to the key, or None if not found.
        """
        return self.data.get(key, None)

    def get_tools(self):
        """
        Return tools in a list for use in the agent.
        """
        return [
            Tool(
                name="GetDataByKey",
                func=self.get_data_by_key,
                description="Retrieve specific data from the JSON file using a key."
            )
        ]
