import soundata
import os
from pathlib import Path

def main():
    print("Initializing soundata for urbansound8k...")
    data_dir = Path("data/UrbanSound8K")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # The dataset initialization. soundata might default to extracting differently 
    # but we will point it to data_home="data/UrbanSound8K"
    dataset = soundata.initialize('urbansound8k', data_home=str(data_dir))
    
    print(f"Downloading dataset to {data_dir}...")
    # This will download and extract the dataset
    dataset.download()
    
    print("Validation:")
    dataset.validate()
    print("Download and validation complete.")

if __name__ == "__main__":
    main()

