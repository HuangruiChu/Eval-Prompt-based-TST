import os
from tqdm import tqdm
import time
import pandas as pd
import openai
from dotenv import load_dotenv
from pdb import set_trace

from utils import gen_prompt

# Set up OpenAI API key
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

model_name = "gpt-3.5-turbo" # or use the frozen one? gpt-3.5-turbo-0301
MAX_TOKENS = 20

def evaluate_prompt(prompt):
    """
    Evaluate a single prompt by calling to OpenAI

    Returns (output, (extra info))
    """

    # Make API call to OpenAI

    # print(f"calling to openai: \"{prompt}\"")
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    try:
        generated_output = completion.choices[0].message.content
        # print(f"\tGenerated: {generated_output}")
        total_tokens = completion.usage.total_tokens

        return generated_output, total_tokens
    except Exception as e:
        print(f"Failed with prompt \"{prompt}\". Exception: {e}")

def continue_prompting_calls(df, input_col, target_style_col, prompt_style) -> bool:
    if df.empty:
        print("Dataframe is empty")
        return False
    
    tmp_row = df.iloc[0]
    input_sentence = tmp_row[input_col]
    target_style = tmp_row[target_style_col]
    tmp_prompt = gen_prompt(input_sentence, target_style, prompt_style=prompt_style)

    print(f"Length of dataset: {len(df)}")    
    continue_proc = ""
    while continue_proc not in ["Y", "N"]:
        continue_proc = input(f"Is this the prompt you expect?\n'{tmp_prompt}'\nEnter Y/N: ")
    if continue_proc == "N":
        return False
    return True

def evaluate_df(df, input_col, target_style_col, dataset_name, prompt_style="zero_shoot", output_dir="outputs"):
    """Call rows of dataframe to OpenAI API
    
    input_col: column name of input sentences
    target_style_col: column name of target style
    """

    if not continue_prompting_calls(df, input_col, target_style_col, prompt_style):
        print("Exiting...")
        return

    output_csv = f"{output_dir}/{dataset_name}-{prompt_style}.csv"
    print(f"Writing output to {output_csv}")

    success = 0
    total_tokens = 0
    generated_outputs = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_sentence = row[input_col]
        target_style = row[target_style_col]
        
        while success <= index:
            try:
                prompt = gen_prompt(input_sentence, target_style, prompt_style=prompt_style)
                generated_output, tokens = evaluate_prompt(prompt)
                generated_outputs.append(generated_output)
                total_tokens += tokens
                pd.DataFrame(generated_outputs).to_csv(output_csv, index=False, header=False)
                success += 1
            except (openai.error.RateLimitError, openai.error.APIError) as e:
                print(e._message)
                set_trace()
            # time.sleep(20) # no longer needed, rate limit was increased...?
    print(f"Total tokens used: {total_tokens}")

    return generated_outputs

# def parallel_evaluate_df(df, input_col, target_style_col, dataset_name, prompt_style="zero_shoot", output_dir="outputs"):

def evaluate_df_parallel(df, input_col, target_style_col, dataset_name, prompt_style="zero_shoot", output_dir="outputs"):
    """Evaluate df in parallel"""

    if not continue_prompting_calls(df, input_col, target_style_col, prompt_style):
        print("Exiting...")
        return

    output_csv = f"{output_dir}/{dataset_name}-{prompt_style}.csv"
    print(f"Writing output to {output_csv}")

    def evaluate_prompt_row(index_row_tuple):
        """From a row in a df
        
        Returns (generated output, original index) of that row
        """
        
        index, row = index_row_tuple
        # print(f"row is type {type(row)} and looks like\n\t{row}")
        input_sentence = row[input_col]
        target_style = row[target_style_col]
        tries = 0
        while tries < 3:
            try:
                prompt = gen_prompt(input_sentence, target_style, prompt_style=prompt_style)
                generated_output, _ = evaluate_prompt(prompt)
                return index, generated_output
            except openai.error as e:
                print(e._message)
                tries += 1
                time.sleep(10)
                # return index, "<ERROR>"
            # except (openai.error.RateLimitError, openai.error.APIError) as e:
            #     print(e._message)
                # set_trace()
    
    #   vanilla df.apply
    # with tqdm(total=len(df)) as pbar:
    #     results = df.apply(evaluate_prompt_row, axis=1, input_col=input_col, target_style_col=target_style_col, prompt_style=prompt_style)
    #     pbar.update(1)
    
    #   pandarallel
    # from pandarallel import pandarallel # import multiprocessing as mp
    # pandarallel.initialize(progress_bar=True)
    # results = df.transpose().parallel_apply(evaluate_prompt_row)

    #   concurrent.futures
    import concurrent.futures
    #   my computer has 12 (check using multiprocessor package)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        # Call the get_completion function for each row in the data list
        generated_index_tuple = list(tqdm(
            executor.map(evaluate_prompt_row, df.iterrows()),
            total=len(df)
        ))

    #   check that outputs are in the right order.
    index_order = [val[0] for val in generated_index_tuple]
    print(index_order)
    if index_order != sorted(index_order):
        print("parallelization outputted in different order!")
        generated_index_tuple.sort(key=lambda x: x[0])
    
    #   save to file
    index = [val[0] for val in generated_index_tuple]
    generated_outputs = [val[1] for val in generated_index_tuple]
    generated_outputs = pd.DataFrame({"output": generated_outputs}, index=index)
    pd.DataFrame(generated_outputs).to_csv(output_csv)
    
    return generated_outputs

if __name__ == "__main__":
    #   test single prompt
    # prompt = gen_prompt("ever since joes has changed hands it 's just gotten worse and worse .", "neg")
    # generated_output = evaluate_prompt(prompt)
    # print(generated_output)

    #   test Yelp dummy
    # df = pd.read_csv("Yelp/yelp_dummy_test.csv", header=None)
    # evaluate_df(df, "yelp_dummy") # DEPRICATED

    #   test GYAFC dummy in sequential
    # df = pd.read_csv("GYAFC/GYAFC_dummy.csv", keep_default_na=False)
    # evaluate_df(df, "input", "target_style", "GYAFC_dummy", prompt_style="zero_shoot")
    #   in parallel
    df = pd.read_csv("GYAFC/GYAFC_dummy.csv", keep_default_na=False)
    df = df.iloc[15:19, :]
    evaluate_df_parallel(df, "input", "target_style", "GYAFC_dummy", prompt_style="zero_shoot")

    #   test GYAFC full (1332)
    # df = pd.read_csv("GYAFC/GYAFC_test.csv", keep_default_na=False)
    # # in parallel
    # evaluate_df_parallel(df, "input", "target_style", "GYAFC_dummy", prompt_style="zero_shoot")

    #   test range(0, 100) of GYAFC
    # df = pd.read_csv("GYAFC/GYAFC_test.csv", keep_default_na=False)
    # df = df.iloc[0:100, :]
    # print(len(df))
    # evaluate_df(df, "input", "target_style", "GYAFC_100", prompt_style="zero_shoot")

    #   test range(100, 200) of GYAFC
    #   in parallel
    # df = pd.read_csv("GYAFC/GYAFC_test.csv", keep_default_na=False)
    # df = df.iloc[100:200, :]
    # print(df)
    # evaluate_df_parallel(df, "input", "target_style", "GYAFC_100-200", prompt_style="zero_shoot")
    
