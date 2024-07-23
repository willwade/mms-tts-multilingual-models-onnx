#!/usr/bin/env python3

import os
import subprocess
import shutil
import collections
import torch
from typing import Any, Dict
import onnx
from huggingface_hub import HfApi, HfFolder
from vits import commons, utils
from vits.models import SynthesizerTrn
from googletrans import Translator

STATE_FILE = "state.txt"
HF_REPO_ID = "willwade/mms-tts-multilingual-models-onnx"  # Replace with your Hugging Face repo ID
SHERPA_ONNX_EXECUTABLE = "/home/ubuntu/sherpa-onnx/build/bin/sherpa-onnx-offline-tts"  # Update this path
VITS_MMS_SCRIPT = "/home/ubuntu/mms-tts-multilingual-models-onnx/vits-mms.py"  # Update this path

def main():
    iso_codes = parse_support_list("support_list.txt")
    processed_iso_codes = load_state()

    for iso_code, language_name in iso_codes.items():
        if iso_code in processed_iso_codes:
            print(f"Skipping {language_name} ({iso_code}), already processed.")
            continue

        print(f"Processing {language_name} ({iso_code})")

        try:
            download_model_files(iso_code)
            generate_model_files(iso_code)
            save_model_files(iso_code)
            translated_sentence = translate_test_sentence("hello everyone this is a test sentence", iso_code)
            if translated_sentence:
                validate_model(iso_code, translated_sentence)
            else:
                print(f"Skipping validation for {iso_code} due to translation failure.")
            push_to_huggingface(iso_code)
            update_state(iso_code)
        except Exception as e:
            print(f"Error processing {iso_code}: {e}")
        finally:
            # Clean up the temporary directory
            tmp_dir = f"tmp/{iso_code}"
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

def parse_support_list(filepath: str) -> Dict[str, str]:
    iso_codes = {}
    with open(filepath, "r", encoding="utf-8") as file:
        # Skip the first line (header)
        next(file)
        for line in file.readlines():
            if line.strip():
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    iso_code, language_name = parts
                    iso_codes[iso_code.strip()] = language_name.strip()
    return iso_codes

def load_state() -> set:
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()

def update_state(iso_code: str):
    with open(STATE_FILE, "a") as f:
        f.write(f"{iso_code}\n")

def download_model_files(iso_code: str):
    base_url = f"https://huggingface.co/facebook/mms-tts/resolve/main/models/{iso_code}"
    files = ["G_100000.pth", "config.json", "vocab.txt"]
    tmp_dir = f"tmp/{iso_code}"
    os.makedirs(tmp_dir, exist_ok=True)
    for file in files:
        result = subprocess.run(["wget", "-q", f"{base_url}/{file}", "-O", f"{tmp_dir}/{file}"])
        if result.returncode != 0:
            raise FileNotFoundError(f"Failed to download {file} for {iso_code}")

def generate_model_files(iso_code: str):
    try:
        tmp_dir = f"tmp/{iso_code}"
        os.environ["PYTHONPATH"] = f"{os.getcwd()}/MMS:{os.getenv('PYTHONPATH', '')}"
        os.environ["PYTHONPATH"] = f"{os.getcwd()}/MMS/vits:{os.getenv('PYTHONPATH', '')}"
        os.environ["lang"] = iso_code
        result = subprocess.run(["python3", VITS_MMS_SCRIPT, tmp_dir, tmp_dir])  # Pass tmp_dir as argument for config and output
        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate model files for {iso_code}")
        
        # Check if model.onnx was created
        if not os.path.isfile(f"{tmp_dir}/model.onnx"):
            raise FileNotFoundError(f"Model file not found: {tmp_dir}/model.onnx")
    except Exception as e:
        raise RuntimeError(f"Failed to generate model files for {iso_code}: {e}")

def save_model_files(iso_code: str):
    tmp_dir = f"tmp/{iso_code}"
    output_dir = f"models/{iso_code}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.move(f"{tmp_dir}/model.onnx", f"{output_dir}/model.onnx")
    shutil.move(f"{tmp_dir}/tokens.txt", f"{output_dir}/tokens.txt")

def translate_test_sentence(sentence: str, target_language: str) -> str:
    translator = Translator()
    try:
        translation = translator.translate(sentence, dest=target_language)
        return translation.text
    except Exception as e:
        print(f"Failed to translate test sentence to {target_language}: {e}")
        return None

def validate_model(iso_code: str, translated_sentence: str):
    output_dir = f"models/{iso_code}"
    model_path = f"{output_dir}/model.onnx"
    tokens_path = f"{output_dir}/tokens.txt"
    wav_output = f"{output_dir}/sample.wav"
    if not os.path.isfile(SHERPA_ONNX_EXECUTABLE):
        raise FileNotFoundError(f"Sherpa-ONNX executable not found: {SHERPA_ONNX_EXECUTABLE}")
    
    result = subprocess.run([
        SHERPA_ONNX_EXECUTABLE,
        f"--vits-model={model_path}",
        f"--vits-tokens={tokens_path}",
        "--debug=1",
        f"--output-filename={wav_output}",
        translated_sentence
    ])
    if result.returncode != 0:
        raise RuntimeError(f"Validation failed for {iso_code}")

    if not os.path.isfile(wav_output):
        raise FileNotFoundError(f"Sample WAV file not generated for {iso_code}")

def push_to_huggingface(iso_code: str):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable not set")

    output_dir = f"models/{iso_code}"
    api = HfApi()
    files_to_upload = [f"{output_dir}/model.onnx", f"{output_dir}/tokens.txt", f"{output_dir}/sample.wav"]
    for file in files_to_upload:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=f"{iso_code}/{os.path.basename(file)}",
            repo_id=HF_REPO_ID,
            token=hf_token
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")