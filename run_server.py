from classifier.server import ClassificationServer
import uvicorn




if __name__ == "__main__":
    server = ClassificationServer()

    uvicorn.run(server, host="127.0.0.1", port=8000)