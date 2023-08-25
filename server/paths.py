import os

store_path = "./store"


def get_project_store() -> str:
    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)
    return store_path


def get_original_image_folder(project_path: str) -> str:
    original_folder_path = os.path.join(project_path, "original")
    if not os.path.exists(original_folder_path):
        os.makedirs(original_folder_path, exist_ok=True)
    return original_folder_path


def get_preprocessed_image_folder(project_path: str) -> str:
    original_folder_path = os.path.join(project_path, "preprocess")
    if not os.path.exists(original_folder_path):
        os.makedirs(original_folder_path, exist_ok=True)
    return original_folder_path


def get_dataset_folder(project_path: str) -> str:
    original_folder_path = os.path.join(project_path, "dataset")
    if not os.path.exists(original_folder_path):
        os.makedirs(original_folder_path, exist_ok=True)
    return original_folder_path


def get_model_output_folder(project_path: str) -> str:
    original_folder_path = os.path.join(project_path, "model_output")
    if not os.path.exists(original_folder_path):
        os.makedirs(original_folder_path, exist_ok=True)
    return original_folder_path
