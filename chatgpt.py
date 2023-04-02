import openai
import requests
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Set up OpenAI API key
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

# Define Hugging Face model and evaluation dataset
model_name = "gpt-3.5-turbo" # or use the frozen one? gpt-3.5-turbo-0301
evaluation_dataset = "your_dataset_url_here" # TODO

# Download evaluation dataset
response = requests.get(evaluation_dataset) # TODO
data = response.json()

# Process and prepare the dataset
# Make sure to preprocess the data according to your model's requirements and evaluation metric

def evaluate_example(example):
    prompt = example["input"]
    expected_output = example["output"]

    # Make API call to OpenAI with the Hugging Face model and prompt
    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        max_tokens=len(expected_output),
        n=1,
        stop=None,
        temperature=1
    )

    generated_output = response.choices[0].text.strip()
    return generated_output, expected_output

# Evaluate the model on the dataset
correct_count = 0
total_count = len(data)

for example in tqdm(data):
    generated_output, expected_output = evaluate_example(example)

    # Check if the generated output matches the expected output
    if generated_output == expected_output:
        correct_count += 1

# Calculate the accuracy
accuracy = correct_count / total_count
print(f"Accuracy: {accuracy:.2%}")

# TODO save results to a file
