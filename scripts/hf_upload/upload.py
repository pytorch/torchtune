import argparse
import os
from huggingface_hub import HfApi, Repository

def add_files_to_repo(api, token, username, repo_name, file_paths):
    """
    Helper function to add files to a repository on Hugging Face Hub.

    Parameters:
    api: Instance of HfApi for API calls.
    token (str): Hugging Face API token.
    username (str): Hugging Face username.
    repo_name (str): Name of the repository on Hugging Face Hub.
    file_paths (list): List of file paths to upload.
    """

    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            api.upload_file(
                token=token,
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=f"{username}/{repo_name}"
            )

    
def upload_to_hf_hub(args):
    """
    Uploads a model, README, and a license file to the Hugging Face Hub.

    Parameters:
    args: Command line arguments.
    """

    # Default license path if not provided
    if not args.license_path:
        args.license_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'LICENSE')

    if not args.readme_path:
        args.readme_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'README.md')

    # Initialize Hugging Face API
    api = HfApi()

    # Create a new private repository on the Hub
    repo_url = api.create_repo(args.repo_name, private=args.private, token=args.hf_token, exist_ok=True)

    # Clone the repository
    repo = Repository(args.repo_name, clone_from=repo_url, use_auth_token=args.hf_token)
    repo.git_pull()

    # Copy files to the repository
    file_paths = [args.model_path, args.readme_path, args.license_path]
    add_files_to_repo(api, args.hf_token, args.hf_username, args.repo_name, file_paths)


    # Commit and push the files to the repository
    repo.commit("Initial commit with model, README, and License")
    repo.push_to_hub()

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Upload a model, README, and License to Hugging Face Hub.")

    # Add arguments
    parser.add_argument("--repo_name", type=str, required=True, help="Name of the repository on Hugging Face Hub.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model file.")
    parser.add_argument("--readme_path", type=str, required=False, help="Path to the README file.")
    parser.add_argument("--hf_username", type=str, required=True, help="Hugging Face username.")
    parser.add_argument("--hf_token", type=str, required=False, help="Hugging Face API token.")
    parser.add_argument("--license_path", type=str, required=False, help="Path to the License file. Defaults to 'LICENSE' in the script directory.", default=None)
    parser.add_argument("--private", action="store_true", help="Set this flag to make the repository private. Omit for a public repository.")

    # Parse arguments
    args = parser.parse_args()

    # Call the upload function
    upload_to_hf_hub(args)
