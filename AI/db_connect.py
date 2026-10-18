import weaviate
# from google.colab import userdata

WEAVIATE_HOST = "32.236.127.40"
WEAVIATE_HTTP_PORT = 8080 # Assuming HTTP port 8080 for Weaviate
WEAVIATE_GRPC_PORT = 50051 # Assuming gRPC port 50051 for Weaviate

# try:
#     # API key for secure access to Weaviate, if required
#     weaviate_api_key = userdata.get("WEAVIATE_API_KEY")
# except userdata.SecretNotFoundError:
#     print("WEAVIATE_API_KEY secret not found. Connecting without API key.")
#     weaviate_api_key = None

def connect_vdb(config):
    print(config)
    try:
        print(f"Connecting to Weaviate at {config['host']}:{config['http_port']} (HTTP) and {config['grpc_port']} (gRPC)...")
        auth_config = None
        # if weaviate_api_key:
        #     auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key)

        # Attempt to connect using the HTTP port first
        weaviate_client = weaviate.connect_to_custom(
            http_host=config["host"],
            http_port=config["http_port"],
            http_secure=config["http_secure"],
            grpc_host=config["host"],
            grpc_port=config["grpc_port"],
            grpc_secure=config["grpc_secure"],
        )

        if weaviate_client.is_live():
            print("Successfully connected to Weaviate (HTTP)!")
            print(f"Weaviate cluster status: {weaviate_client.get_meta()['version']}") # Access version as a dictionary key

            # Use client.collections.list_all() for the new client
            collections = weaviate_client.collections.list_all()
            collection_names = [c.name for c in collections.values()]
            print(f"Available collections (Weaviate): {collection_names}")
            return {
                "weaviate_client": weaviate_client,
                "collections": collections,
                "collection_names": collection_names,
            }
        else:
            print("Failed to connect to Weaviate (HTTP).")
            weaviate_client = None
            return weaviate_client

    except Exception as e:
        print(f"Failed to connect to Weaviate: {e}")
        weaviate_client = None
        return weaviate_client
