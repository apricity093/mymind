"""按固定官方 revision 下载完整基座和 tokenizer。"""
import argparse
import os
from huggingface_hub import snapshot_download
from agents.customer_policy.configuration import load_configuration
from rl.runtime import write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root, config, output = load_configuration(args.config)
    destination = root / config["model_dir"]
    snapshot_download(repo_id=config["model_id"], revision=config["model_revision"],
                      local_dir=destination, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
                      max_workers=3)
    write_json(destination / "base_revision.json", {"repo_id": config["model_id"],
        "revision": config["model_revision"], "download_endpoint": os.getenv("HF_ENDPOINT", "https://huggingface.co")})
