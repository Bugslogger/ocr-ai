"""Runtime configuration. Importing this module performs no I/O."""
import os

Qconfig = {"max_new_tokens": 8000, "temperature": 0.0}
Oconfig = {"max_length": 15000, "image_size": 1024, "dpi": 350}
db = {
    "host": os.getenv("WEAVIATE_HOST", "32.236.127.40"),
    "http_port": int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
    "grpc_port": int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
    "http_secure": os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true",
    "grpc_secure": os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true",
}
