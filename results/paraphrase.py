"""Frozen Paraphraser"""

import argparse
import logging
import sys
import torch
from pdb import set_trace
import pandas as pd
from tqdm import tqdm

from style_paraphrase.inference_utils import GPT2Generator

# make sure to call this before running program
# $ export CUDA_VISIBLE_DEVICES=1

parser = argparse.ArgumentParser()
parser.add_argument("path_input")
parser.add_argument("path_output")
parser.add_argument("-s", type=int, default=0)
parser.add_argument("-e", type=int, default=None)
args = parser.parse_args()

path_input = args.path_input
path_output = args.path_output

df = pd.read_csv(path_input)
start = args.s
if args.e:
    end = args.e
else:
    end = len(df)
df = df.iloc[start:end, :]
print(df)
set_trace()

print('loading model')
paraphraser = GPT2Generator("paraphraser_gpt2_large", upper_length="same_5")
paraphraser.modify_p(top_p=0)

def normalize_input(index_row_tuple):
    """Return normalized version of text"""
    index, row = index_row_tuple
    try:
        greedy_decoding = paraphraser.generate(row["input"])
    except Exception as e:
        set_trace()
    return index, greedy_decoding

import concurrent.futures
# multiprocessing.cpu_count() says 24, but experimenting manually, it can't work with large number of workers. maybe memory runs out?
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    # Call the get_completion function for each row in the data list
    generated_index_tuple = list(tqdm(
        executor.map(normalize_input, df.iterrows()),
        total=len(df)
    ))

index_order = [val[0] for val in generated_index_tuple]
print(index_order)
if index_order != sorted(index_order):
    print("parallelization outputted in different order!")
    generated_index_tuple.sort(key=lambda x: x[0])

index = [val[0] for val in generated_index_tuple]
generated_outputs = [val[1] for val in generated_index_tuple]
generated_outputs = pd.DataFrame({"output": generated_outputs}, index=index)
pd.DataFrame(generated_outputs).to_csv(path_output)
