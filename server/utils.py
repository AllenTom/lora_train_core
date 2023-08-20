import hashlib


def read_sha256_from_file_path(filepath:str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()