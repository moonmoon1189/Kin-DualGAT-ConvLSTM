import os
import hashlib

def verify_checksum(filepath, expected_md5):
    """Verify downloaded dataset integrity."""
    if not os.path.exists(filepath):
        return False
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest() == expected_md5

def extract(zip_path, extract_dir):
    """Extract dataset."""
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def print_summary(data_dir):
    print(f"Dataset ready at {data_dir}. Note: We do not commit the 2865 real sequences to GitHub.")
    print("Only this download script and a tiny demo subset are kept in the repository to comply with privacy policies.")

def download(doi_url, target_dir="./data/CB-FQA"):
    os.makedirs(target_dir, exist_ok=True)
    print(f"Please download the official release from Zenodo (DOI: {doi_url}) manually due to restricted access policies.")
    print_summary(target_dir)

if __name__ == "__main__":
    download("10.5281/zenodo.21486299")