import requests




class ClassificationAPI:
    def __init__(self):
        self.server_url: str = "http://127.0.0.1:8000/process_text/"


    def classify_text(self, text: str) -> dict:
        if not isinstance(text, str):
            return {'error': f"text must be a string but got {type(text)}"}

        # Prepare Request JSON Data
        request_json: dict = {'text': text}

        # Check if server is running
        try: 
            response = requests.post(self.server_url, json=request_json)
        except Exception as e:
            return {'error': str(e)}

        # Check if response is ok
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'{response.status_code}: {response.text}'}