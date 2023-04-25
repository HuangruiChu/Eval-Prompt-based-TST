import os
from tqdm import tqdm
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
        return generated_output
    except Exception as e:
        print(f"Failed with prompt \"{prompt}\". Exception: {e}")


def eval_df(df, dataset_name, input_col, target_style_col, prompt_style="zero_shoot", output_dir="outputs"):
    """Call rows of dataframe to OpenAI API
    
    input_col: column name of input sentences
    target_style_col: column name of target style
    """
    
    output_csv = f"{output_dir}/{dataset_name}-{prompt_style}.csv"
    print(f"Writing to {output_csv}")



    success = 0
    generated_outputs = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_sentence = row[input_col]
        target_style = row[target_style_col]
        
        while success <= index:
            try:
                prompt = gen_prompt(input_sentence, target_style, prompt_style=prompt_style)
                generated_outputs.append(evaluate_prompt(prompt))
                pd.DataFrame(generated_outputs).to_csv(output_csv, index=False, header=False)
                success += 1
            except (openai.error.RateLimitError, openai.error.APIError) as e:
                print(e._message)
                set_trace()
            # time.sleep(20) # no longer needed, rate limit was increased...?


if __name__ == "__main__":
    #   test single prompt
    # prompt = gen_prompt("ever since joes has changed hands it 's just gotten worse and worse .", "neg")
    # generated_output = evaluate_prompt(prompt)
    # print(generated_output)

    #   test Yelp dummy
    # df = pd.read_csv("Yelp/yelp_dummy_test.csv", header=None)
    # evaluate_df(df, "yelp_dummy") # DEPRICATED

    #   test GYAFC dummy
    df = pd.read_csv("GYAFC/GYAFC_dummy.csv", keep_default_na=False)
    eval_df(df, "GYAFC_dummy", "input", "target_style", prompt_style="zero_shoot")

    #   test GYAFC full (1332)
    # df = pd.read_csv("GYAFC/informal_to_formal.csv", keep_default_na=False)
