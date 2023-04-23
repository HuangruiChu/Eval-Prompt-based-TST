import os
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
    """
    # Make API call to OpenAI

    print(f"calling to openai: \"{prompt}\"")
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    try:
        generated_output = completion.choices[0].message.content
        print(f"\tGenerated: {generated_output}")
        return generated_output
    except Exception as e:
        print(f"Failed with prompt \"{prompt}\". Exception: {e}")

# current pace: 3 calls per minute (rate limit is 3 per minute)
def evaluate_df(df, dataset_name, prompt_style="zero_shoot", output_dir="outputs"):
    """
    Iteratively call rows of the dataframe to the OpenAI API
    """
    print(f"Using prompt style: {prompt_style}")
    generated_outputs = []

    success = 0
    for index, (label, input_sentence, expected_output) in df.iterrows():
        while success <= index:
            try:
                prompt = gen_prompt(input_sentence, label, prompt_style=prompt_style)
                generated_outputs.append(evaluate_prompt(prompt))
                pd.DataFrame(generated_outputs).to_csv(f"{output_dir}/{dataset_name}-{prompt_style}.csv", index=False, header=False)
                success += 1
            except (openai.error.RateLimitError, openai.error.APIError) as e:
                print(e._message)
                # time.sleep(20)
            time.sleep(20)

if __name__ == "__main__":
    #   test single prompt
    # prompt = gen_prompt("ever since joes has changed hands it 's just gotten worse and worse .", "neg")
    # generated_output = evaluate_prompt(prompt)
    # print(generated_output)

    #   test dataset
    filepath = "Yelp/yelp_dummy_test.csv"
    df = pd.read_csv(filepath, header=None)
    evaluate_df(df, "yelp_dummy")
