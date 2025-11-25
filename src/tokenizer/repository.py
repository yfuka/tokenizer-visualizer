import os
import shutil
from typing import List, Dict
from transformers import AutoTokenizer

class TokenizerRepository:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.hf_dir = os.path.join(base_dir, "huggingface")
        self.uploads_dir = os.path.join(base_dir, "uploads")
        
        os.makedirs(self.hf_dir, exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)

    def get_available_models(self) -> Dict[str, str]:
        """
        Returns a dictionary of {display_name: path} for all available models.
        """
        models = {}
        
        # 1. Scan Hugging Face downloads
        if os.path.exists(self.hf_dir):
            for item in os.listdir(self.hf_dir):
                full_path = os.path.join(self.hf_dir, item)
                if os.path.isdir(full_path):
                    # Use the directory name as the model ID (replace _ with / if we sanitized it)
                    # For simplicity, we'll just use the directory name as display
                    display_name = item.replace("___", "/") 
                    models[f"HF: {display_name}"] = full_path

        # 2. Scan Uploads
        if os.path.exists(self.uploads_dir):
            for item in os.listdir(self.uploads_dir):
                full_path = os.path.join(self.uploads_dir, item)
                
                # Support legacy single files
                if os.path.isfile(full_path) and item.endswith(".json"):
                    models[f"Local File: {item}"] = full_path
                
                # Support directory-based models (new standard)
                elif os.path.isdir(full_path):
                    models[f"Local Dir: {item}"] = full_path
                    
        return models

    def download_model(self, repo_id: str) -> str:
        """
        Downloads a tokenizer from Hugging Face and saves it locally.
        Returns the path to the saved model.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            
            # Sanitize repo_id for directory name
            safe_name = repo_id.replace("/", "___")
            save_path = os.path.join(self.hf_dir, safe_name)
            
            tokenizer.save_pretrained(save_path)
            return save_path
        except Exception as e:
            raise RuntimeError(f"Failed to download model '{repo_id}': {e}")

    def save_uploaded_model_batch(self, files, model_name: str) -> str:
        """
        Saves a batch of uploaded files to a directory in uploads.
        Returns the path to the saved directory.
        """
        # Sanitize model name for directory
        safe_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_')).strip()
        if not safe_name:
            raise ValueError("Invalid model name")
            
        save_dir = os.path.join(self.uploads_dir, safe_name)
        os.makedirs(save_dir, exist_ok=True)
        
        for uploaded_file in files:
            file_path = os.path.join(save_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
        return save_dir

    def save_uploaded_model(self, file_obj, filename: str) -> str:
        """
        Saves an uploaded file to the uploads directory.
        Returns the path to the saved file.
        """
        save_path = os.path.join(self.uploads_dir, filename)
        with open(save_path, "wb") as f:
            f.write(file_obj.getbuffer())
        return save_path
