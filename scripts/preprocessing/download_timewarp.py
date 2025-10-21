import os

from huggingface_hub import hf_hub_download, list_repo_files


def download_files(repo_id, subfolder, local_dir):
    all_files = list_repo_files(repo_id, repo_type="dataset")
    files_to_download = [file for file in all_files if file.startswith(subfolder)]

    for i, file in enumerate(files_to_download):
        local_path = os.path.join(local_dir)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        hf_hub_download(repo_id, filename=file, local_dir=local_dir, repo_type="dataset")
        print(f"{i} / {len(files_to_download)} Downloaded {file} to {local_path}")


def main(args):

    repo_id = "microsoft/timewarp"
    subfolder = args.dataset
    local_dir = "storage/timewarp"

    download_files(repo_id, subfolder, local_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="4AA-large")
    args = parser.parse_args()

    main(args)
