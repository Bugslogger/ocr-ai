"""Optional Weaviate connection; callers own and close the returned client."""
def connect_vdb(config=None):
    import weaviate
    from .config import db

    settings = dict(db if config is None else config)
    client = weaviate.connect_to_custom(
        http_host=settings["host"], http_port=settings["http_port"],
        http_secure=settings["http_secure"], grpc_host=settings["host"],
        grpc_port=settings["grpc_port"], grpc_secure=settings["grpc_secure"],
    )
    try:
        if not client.is_ready():
            raise ConnectionError("Weaviate is not ready")
        collections = client.collections.list_all()
        return {"weaviate_client": client, "collections": collections,
                "collection_names": list(collections)}
    except Exception:
        client.close()
        raise
