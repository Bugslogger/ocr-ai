from db_connect import connect_vdb
from config import db
from utils import extract_doc

# models
from model import download_ocr, download_qwen

# variables
model_config = None

# connect to the vector database
response = connect_vdb(db)
print(response)

# dowload models
def download_models():
    if not model_config:
        try:
            Omodel_config = download_ocr()
            Qmodel_config = download_qwen()
            return {
                "ocr": Omodel_config,
                "qwen": Qmodel_config,
            }
        except Exception as e:
            return {"error": True, "message": e}
    else:
        print("Model already exists...")
        return model_config

model_config = download_models()

def doc_pipeline():
    extract_doc() 
