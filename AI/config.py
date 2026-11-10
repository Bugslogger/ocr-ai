"""Database configuration and compatibility imports for model settings."""
import os

from .model_config import Qconfig, Qruntime, Oconfig

db = {
    "host": os.getenv("WEAVIATE_HOST", "32.236.127.40"),
    "http_port": int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
    "grpc_port": int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
    "http_secure": os.getenv("WEAVIATE_HTTP_SECURE", "false").lower() == "true",
    "grpc_secure": os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true",
}
