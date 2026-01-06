```python
"""
This script sends a GET request to the GitHub API and prints the JSON response.

Author: [Your Name]
Date: [Today's Date]
"""

import requests

def fetch_github_api_data():
    """
    Sends a GET request to the GitHub API and returns the JSON response.

    Returns:
        dict: The JSON response from the GitHub API.
    """
    url = requests.get('https://api.github.com')
    data = url.json()
    return data

def main():
    """
    The main function that prints the JSON response from the GitHub API.

    Returns:
        None
    """
    data = fetch_github_api_data()
    print(data)

if __name__ == "__main__":
    """
    This block is executed when the script is run directly.

    Returns:
        None
    """
    main()
```