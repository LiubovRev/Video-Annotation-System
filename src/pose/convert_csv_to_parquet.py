import pandas as pd
from pathlib import Path

def convert_processed_csv_to_parquet(projects, directory):
    for project in projects:
        base_path = Path(f"/home/liubov/Bureau/new/{project}/{directory}")
        csv_file = base_path / "processed_data.csv"
        parquet_file = base_path / f"processed_data__{project}.parquet"

        if csv_file.exists():
            print(f"Processing: {csv_file}")
            try:
                # Read CSV and save as Parquet
                df = pd.read_csv(csv_file)
                df.to_parquet(parquet_file, engine="pyarrow", index=False)
                print(f"Saved Parquet: {parquet_file}")
            except Exception as e:
                print(f" Error processing {csv_file}: {e}")
        else:
            print(f" File not found: {csv_file}")

if __name__ == "__main__":
    projects = [
#         '11-1-2024_#9_INDIVIDUAL_[18]',
#         '16-5-2024_20_INDIVIDUAL_[18]',
#         '23-5-2024_#20_INDIVIDUAL_[14]',
#         '14-3-2024_#15_INDIVIDUAL_[18]',
#         '11-1-2024_#7_INDIVIDUAL_[14]',
#         '8-5-2024_#18_INDIVIDUAL_[12]',
        '10-1-2024_#6_INDIVIDUAL_[15]'
    ]

    directory = 'PosesDir'
    convert_processed_csv_to_parquet(projects, directory)
