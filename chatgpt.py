import os
# import csv
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

    # response = openai.Completion.create(
    #     model=model_name,
    #     prompt=prompt,
    #     max_tokens=MAX_TOKENS,
    #     n=1,
    #     stop=None,
    # )
    set_trace()
    print(f"calling to openai: \"{prompt}\"")
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    try:
        generated_output = completion.choices[0].message.content
        print(f"generated: {generated_output}")
        return generated_output
    except Exception as e:
        print(f"Failed with prompt \"{prompt}\". Exception: {e}")

def evaluate_df(df, prompt_style="zero_shoot"):
    """
    Iteratively call rows of the dataframe to the OpenAI API
    """
    generated_outputs = []
    for index, (label, input_sentence, expected_output) in df.iterrows(): 
        prompt = gen_prompt(input_sentence, label, prompt_style=prompt_style)
        generated_outputs.append(evaluate_prompt(prompt))
    

    

# def get_test_dataset(filepath):

#     # with open(filepath) as f:
#     #     raw_data = csv.reader(f)
    
#     return raw_data

#     # prompt_dic = {}
#     # prompt_dic["zero_shoot"] = "prompt1"
#     # outputfile = open('yelp_dummy_{}.csv'.format(prompt_dic[prompt_style]), 'w', newline='',encoding='UTF8')
#     # writer = csv.writer(outputfile)

if __name__ == "__main__":
    #   test single prompt
    prompt = gen_prompt("ever since joes has changed hands it 's just gotten worse and worse .", "neg")
    generated_output = evaluate_prompt(prompt)
    print(generated_output)

    #   test dataset
    # filepath = "Yelp/yelp_dummy_test.csv"
    # df = pd.read_csv(filepath, header=None)
    # evaluate_df(df)