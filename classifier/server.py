from fastapi import FastAPI
from torch import load
import os

from .model import BertClassifier




class ClassificationServer(FastAPI):
    def __init__(self):
        super().__init__()
        # Set up Classifier Model
        self.model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models', 'Classifier_32BS_1024HS_2e-05LR_1717333910', 'Classifier_32BS_1024HS_2e-05LR.pt')
        self.model: BertClassifier = BertClassifier()

        # Load Classifier Weights
        self.model.load_state_dict(load(self.model_path, map_location=next(self.model.parameters()).device), strict=True)
        self.model.eval()

        # Add API Routes
        self.add_routes()

    
    def add_routes(self):
        @self.post("/process_text/")
        async def process_data(data: dict) -> dict:
            if not isinstance(data, dict):
                return {"error": f"data must be a dictionary but got {type(data)}"}
            try:
                text: str = data['text']
                labels: list[str] = self.model.predict(text)
                return {"labels": labels}
            except Exception as e:
                return {"error": str(e)}
