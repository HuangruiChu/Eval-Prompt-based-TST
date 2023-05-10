import pandas as pd
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path_csv")
    parser.add_argument("target_style")
    parser.add_argument("--index_col", default="index")
    
    args = parser.parse_args()
    
    path_csv = args.path_csv
    target_style = args.target_style
    index_col = args.index_col
    
    df = pd.read_csv(path_csv, index_col=index_col)
    new_col = pd.Series([target_style] * len(df))
    df.insert(0, "target_style", new_col)
    # df.rename(columns={"output":"input"})

    df.to_csv(path_csv, index=False)
