# Paper Abstract Text Classification

## Introduction

This repository provides a server-client setup for text classification using a pretrained BERT model. The server handles the classification logic, while the client sends text for classification and retrieves the results. The future improvements section lists potential enhancements to the system.

## Getting Started

Follow these instructions to set up the environment, install dependencies, and run the server and client.

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/georgezampoukis/paper_abstract_text_classification.git
    cd paper_abstract_text_classification
    ```

2. **Initialize a new Python environment:**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download Model Weights:**
    
    - 4.1 [Model Weights](https://we.tl/t-1GBH9Gr7kc)
    - 4.2 Extract the zipped folder into classifier/trained_models
    - 4.3 The folder structure should look like this: 
    - 4.4 classifier\trained_models\Classifier_32BS_1024HS_2e-05LR_1717333910\Classifier_32BS_1024HS_2e-05LR.pt

### Running the Server

1. **Start the Classification Server:**

    ```bash
    python run_server.py
    ```

    - Appropriate INFO: messages will appear when the server is ready.

### Running the Client

1. **Perform classification on a sample abstract text:**

    ```bash
    python run_client.py
    ```

## Future Improvements

1. **Custom Transformer Encoder:**
   - Replace the pretrained BERT model with a custom Transformer Encoder and train it from scratch.

2. **Experiment with Encoder Architectures:**
   - Try different encoder architectures to produce a richer representation vector.

3. **Model Optimization:**
   - Convert models to ONNX and fine-tune with TensorRT for improved performance.

4. **Dataset Expansion:**
   - Add more text samples to the dataset to enhance model training.

5. **Tokenizer Experiments:**
   - Experiment with different tokenizers to find the most effective one for this classification task.
