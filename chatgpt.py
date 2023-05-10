import os
import argparse
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

def evaluate_df(df, input_col, target_style_col, output_name, prompt_style="zero_shoot", output_dir="outputs"):
    """Call rows of dataframe to OpenAI API sequentially
    
    input_col: column name of input sentences
    target_style_col: column name of target style
    """

    if not continue_prompting_calls(df, input_col, target_style_col, prompt_style):
        print("Exiting...")
        return

    output_csv = f"{output_dir}/{output_name}-{prompt_style}.csv"
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


def evaluate_df_parallel(df, input_col, target_style_col, output_name, prompt_style, output_dir):
    """Evaluate df in parallel"""

    if not continue_prompting_calls(df, input_col, target_style_col, prompt_style):
        print("Exiting...")
        return

    output_csv = f"{output_dir}/{output_name}-{prompt_style}.csv"
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
            except (openai.error.RateLimitError, openai.error.APIError) as e:
                print(e._message)
                tries += 1
                time.sleep(10)
                # return index, "<ERROR>"
            # except (openai.error.RateLimitError, openai.error.APIError) as e:
            #     print(e._message)
                # set_trace()

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

    parser = argparse.ArgumentParser(description='Process test set through OpenAI API.')
    parser.add_argument('test_set_path', type=str, help='path to csv file containing test set')
    parser.add_argument('output_name', type=str, help='name of output')
    parser.add_argument('--input_col', type=str, help='name of input column', default="input")
    parser.add_argument('--target_style_col', type=str, help='name of target style column', default="target_style")
    parser.add_argument('--prompt_style', type=str, help='prompt style', default="zero_shoot")
    parser.add_argument('--start', type=int, help='starting index', default=0)
    parser.add_argument('--end', type=int, help='ending index', default=None)
    parser.add_argument('--output_dir', type=str, help='path to directory where we will create the new file', default="outputs/yelp")

    args = parser.parse_args()

    test_set_path = args.test_set_path
    output_name = args.output_name
    input_col = args.input_col
    target_style_col = args.target_style_col
    prompt_style = args.prompt_style
    output_dir = args.output_dir

    df = pd.read_csv(test_set_path, keep_default_na=False)

    start = args.start
    end = args.end

    df = df.iloc[start:end, :]
    print(df)

    evaluate_df_parallel(
        df=df,
        input_col=input_col,
        target_style_col=target_style_col,
        output_name=output_name,
        prompt_style=prompt_style,
        output_dir=output_dir,
    )
