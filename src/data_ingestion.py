import os
import csv
import urllib.request
import urllib.error
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATASETS = {
    "ulb": {
        "url": "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
        "filename": "raw_transactions_ulb.csv"
    },
    "banksim": {
        "url": "https://raw.githubusercontent.com/rupakroy/fraud-detection-dataset/master/fraud_data.csv",
        "filename": "raw_transactions_banksim.csv"
    },
    "ieee": {
        "url": "https://raw.githubusercontent.com/jcarroll/fraud-detection/master/data/train_transaction.csv",
        "filename": "raw_transactions_ieee.csv"
    }
}

# Resolve target data directory relative to this script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))

def find_class_column(header):
    """
    Tries to find target column index based on common fraud label names.
    """
    target_names = ["class", "isfraud", "fraud", "label"]
    header_lower = [col.strip().lower() for col in header]
    for target in target_names:
        if target in header_lower:
            return header_lower.index(target)
    # Default fallback to the last column
    return len(header) - 1

def analyze_csv(file_path):
    """
    Reads the CSV file, calculates its shape and logs the class imbalance.
    Uses only Python standard library to ensure environment independence.
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found for analysis: {file_path}")
        return

    logger.info(f"Analyzing downloaded file: {os.path.basename(file_path)}")
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                logger.warning(f"Empty CSV file encountered: {file_path}")
                return

            num_cols = len(header)
            class_col_idx = find_class_column(header)
            class_col_name = header[class_col_idx] if class_col_idx < len(header) else "Unknown"
            
            row_count = 0
            class_counts = {"0": 0, "1": 0}
            
            for row in reader:
                if not row:
                    continue
                row_count += 1
                if class_col_idx < len(row):
                    val = row[class_col_idx].strip()
                    # Standardize binary outputs
                    if val in ("1", "1.0", "yes", "true", "True"):
                        class_counts["1"] += 1
                    elif val in ("0", "0.0", "no", "false", "False"):
                        class_counts["0"] += 1
                    else:
                        class_counts[val] = class_counts.get(val, 0) + 1

            logger.info(f"Dataset Shape: Rows = {row_count}, Columns = {num_cols}")
            logger.info(f"Detected Class Column: '{class_col_name}' at index {class_col_idx}")
            logger.info(f"Class imbalance distribution: {dict(class_counts)}")
            
            # Print directly to stdout as requested by the user
            print(f"\n=== Analysis for {os.path.basename(file_path)} ===")
            print(f"Shape: ({row_count}, {num_cols})")
            print(f"Target Column Name: '{class_col_name}'")
            print(f"Class Imbalance Counts: {dict(class_counts)}")
            print("=" * (40 + len(os.path.basename(file_path))) + "\n")

    except Exception as e:
        logger.error(f"Error during CSV analysis for {file_path}: {e}")

def download_datasets(data_dir=DATA_DIR):
    """
    Downloads the three distinct datasets safely. Creates target folder if missing.
    """
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Target data directory verified at: {data_dir}")

    for name, info in DATASETS.items():
        url = info["url"]
        filename = info["filename"]
        dest_path = os.path.join(data_dir, filename)

        logger.info(f"Downloading {name.upper()} dataset from: {url}")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(dest_path, 'wb') as out_file:
                    # Chunks of 1MB to efficiently stream large files
                    chunk_size = 1024 * 1024
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
            logger.info(f"Saved: {filename}")
            
            # Run analysis post-download
            analyze_csv(dest_path)
            
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP Error {e.code} for {name.upper()} from {url}: {e.reason}")
        except urllib.error.URLError as e:
            logger.error(f"URL Error for {name.upper()} from {url}: {e.reason}")
        except Exception as e:
            logger.error(f"Failed download/processing for {name.upper()}: {e}")

if __name__ == "__main__":
    download_datasets()
