import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

def classify_sentiment(pred_texts):
    """
    Classify list of texts using a finetuned RoBERTa-Large (SiEBERT)

    {0: "NEGATIVE", 1: "POSITIVE"}
    """

    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)

    class SimpleDataset:
        def __init__(self, tokenized_texts):
            self.tokenized_texts = tokenized_texts
        
        def __len__(self):
            return len(self.tokenized_texts["input_ids"])
        
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.tokenized_texts.items()}


    tokenized_texts = tokenizer(pred_texts,truncation=True,padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)

    predictions = trainer.predict(pred_dataset)

    preds = predictions.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)
    
    return preds, labels, scores

def score_sentiment(generated, target_class="NEGATIVE", output_file=None):
    """Give the fraction and percentage of successful sentiment transfers
    
    Assumes all of the generated sentences are trying to be the target_class
    
    {0: "NEGATIVE", 1: "POSITIVE"}
    """

    assert target_class in ["NEGATIVE", "POSITIVE"]

    preds, _, _ = classify_sentiment(generated)
    
    total = len(preds)
    score = sum(preds)
    if target_class == "NEGATIVE":
        score = total - sum(preds)

    if output_file:
        if target_class == "NEGATIVE":
            expected_scores = total * [0]
        else:
            expected_scores = total * [1]
        individual_scores = [classified == expected for (classified, expected) in zip(preds, expected_scores)]
        print(f"Saving in new file {output_file}")
        pd.DataFrame(individual_scores).to_csv(output_file, index=False, header=None)

    return f"{(score)/total:.2f} ({score}/{total})"


if __name__ == "__main__":
    #   try sentiment classifier on single list
    # pred_texts = ["ever since joes has changed hands it 's just gotten worse and worse .", "I'm so happy today, I want to cry happy tears. I love everything!"]
    # print(classify_sentiment(pred_texts))

    #   try sentiment classifier on dummy dataset
    df = pd.read_csv("outputs/yelp_dummy-zero_shoot.csv", header=None)
    generated = list(df[0])
    print(score_sentiment(generated, output_file="results/yelp_dummy_score.csv"))
    
