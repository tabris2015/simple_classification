import pandas as pd
def csv_to_buckets(csv_file: str, out_dir: str, n_buckets: int = 3):
    """Read a csv file with the target data and copy the images to
    folders for classification data loading"""
    data = pd.read_csv(csv_file, header=0)
    print(data.describe())



if __name__ == "__main__":
    csv_file = "/home/pepe/dev/datasets/dataset/target4.csv"

    csv_to_buckets(csv_file, "/tmp")
