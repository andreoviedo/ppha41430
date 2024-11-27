import requests
import json
import pandas as pd

def get_api_data(url, params=None, headers=None):
    """
    Fetch data from an API endpoint and return as JSON
    
    Args:
        url (str): The API endpoint URL
        params (dict): Optional query parameters
        headers (dict): Optional request headers
    
    Returns:
        df: DataFrame containing the API response data
    """
    try:
        # If params contains an encoded SQL query, append it directly to URL
        if params and 'sql' in params:
            sql_query = params.pop('sql')
            url = f"{url}?sql={sql_query}"
        
        # Make GET request to API with remaining params
        response = requests.get(url, params=params, headers=headers)
        
        # Raise exception for bad status codes
        response.raise_for_status()
        
        # Parse JSON data
        json_data = response.json()
        
        # If this is a SQL query response, extract the 'rows' element
        if isinstance(json_data, dict) and 'rows' in json_data:
            df = pd.DataFrame(json_data['rows'])
        else:
            df = pd.DataFrame(json_data)
            
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching API data: {e}")
        return None

def encode_query(query):
    """
    Encode a PostgreSQL query string into percent-encoded format
    
    Args:
        query (str): The PostgreSQL query to encode
        
    Returns:
        str: The percent-encoded query string
    """
    try:
        from urllib.parse import quote
        # Replace line breaks with spaces before encoding
        query = ' '.join(query.splitlines())
        encoded_query = quote(query)
        return encoded_query
    except Exception as e:
        print(f"Error encoding query: {e}")
        return None

