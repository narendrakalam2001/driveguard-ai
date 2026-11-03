from getpass import getpass
from huggingface_hub import HfApi

repo = "narendrakalam2001/driveguard-ai-models"   # <- username badlo
api = HfApi(token=getpass("HF WRITE token: "))   # paste karo, screen pe nahi dikhega
api.create_repo(repo, repo_type="model", exist_ok=True)  # public repo
for name in ("yolov8_vehicle.pt", "yolov8_plate.pt"):
    api.upload_file(path_or_fileobj=f"models/{name}", path_in_repo=name, repo_id=repo)
    print("uploaded", name)