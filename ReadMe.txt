-------- Running Instructions --------

1. Initialize a new Python Environment
2. Install Environment Dependencies with 'pip install -r requirements.txt'
3. Run the 'run_server.py' to initialize the Classification Server
5. Run the 'run_client.py' to perform classification on a sample abstract text


-------- Future Improvements --------

1. Replace the pretrained Bert model with a custom Transformer Encoder to train from scratch
2. Experiment with different Encoder architectures to produce a richer representation vector
3. Convert models to ONNX and finetune with TensorRT for better performance
4. Add more text samples to the dataset
5. Experiment with different Tokennizers