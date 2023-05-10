import pandas as pd
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path_csv")
    parser.add_argument("--index_col", default="index")
    
    args = parser.parse_args()
    
    path_csv = args.path_csv
    index_col = args.index_col
    
    df = pd.read_csv(path_csv, index_col=index_col)
    df.to_csv(path_csv, index=False, header=False)
