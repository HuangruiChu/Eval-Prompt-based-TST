import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import AutoModelForCausalLM
import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from pdb import set_trace

class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

def classify_sentiment(pred_texts):
    """Classify list of texts using a finetuned RoBERTa-Large (SiEBERT)

    {0: "NEGATIVE", 1: "POSITIVE"}
    """

    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)

    tokenized_texts = tokenizer(pred_texts,truncation=True,padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)

    predictions = trainer.predict(pred_dataset)

    preds = predictions.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)
    
    return preds, labels, scores

def classify_formality(pred_texts):
    """Give the fraction and percentage of successful sentiment transfers

    {0: "informal", 1: "formal"}
    """
    
    model_name = "s-nlp/roberta-base-formality-ranker"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)

    tokenized_texts = tokenizer(pred_texts,truncation=True,padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)

    predictions = trainer.predict(pred_dataset)

    # outputs = model(**tokenized_texts)
    # predictions = torch.argmax(outputs.logits, dim=1)

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
    
def score_formality(generated, target_class="formal", output_file=None):
    """Give the fraction and percentage of successful formality transfers
    
    Assumes all of the generated sentences are trying to be the target_class

    {0: "informal", 1: "formal"}
    """

    assert target_class in ["informal", "formal"]

    preds, _, _ = classify_formality(generated)

    total = len(preds)
    score = sum(preds)
    if target_class == "informal":
        score = total - sum(preds)
    
    if output_file:
        if target_class == "informal":
            expected_scores = total * [0]
        else:
            expected_scores = total * [1]
        individual_scores = [classified == expected for (classified, expected) in zip(preds, expected_scores)]
        print(f"Saving in new file {output_file}")
        pd.DataFrame(individual_scores).to_csv(output_file, index=False, header=None)
    
    print(f"{(score)/total:.2f} ({score}/{total})")
    return f"{(score)/total:.2f} ({score}/{total})"

def score_BLEU(generated_list, refs_list, output_file=None):
    """Give the average of the BLEU score
     
    Uses SacreBLEU
    """

    bleu = BLEU()

    # individual_scores = []
    # for generated, refs in zip(generated_list, refs_list):
    
    bleu_score = bleu.corpus_score(generated_list, refs_list)

    set_trace()
    return bleu_score.score

    # if output_file:
    #     print(f"Saving in new file {output_file}")
    #     pd.DataFrame(individual_scores).to_csv(output_file, index=False, header=None)

def score_ppl(generated_list):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def score(sentence):
        inputs = tokenizer(sentence, return_tensors = "pt")
        loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        ppl = torch.exp(loss)
        return ppl.item()
    
    ppl_list = [score(generated) for generated in tqdm(generated_list)]
    return sum(ppl_list)/len(ppl_list)

def remove_end_quote(x):
    if x[0][-1] == '"':
        return x[0][:-1]
    else:
        return x[0]

if __name__ == "__main__":
    #   try sentiment classifier on single list
    # pred_texts = ["ever since joes has changed hands it 's just gotten worse and worse .", "I'm so happy today, I want to cry happy tears. I love everything!"]
    # print(classify_sentiment(pred_texts))

    #   try sentiment classifier on dummy dataset
    # df = pd.read_csv("outputs/yelp_dummy-zero_shoot.csv", header=None)
    # df = df.apply(remove_end_quote, axis=1)

    # generated = list(df[0])
    # print(score_sentiment(generated, output_file="results/yelp_dummy_score.csv"))

    #   try classify formality
    # generated = ['Therefore, what is the implication if both parties involved are engaging in a rebound relationship?', 'Wishing you the best of luck in your pursuit of the ideal candidate.', 'What is the underlying motivation driving individuals to pursue unobtainable individuals and knowingly desire that which they should not?', 'Do you have a proclivity for engaging in contentious debates and disputes?', 'If that is your definitive decision, then that would be the recommended approach.']
    # print(classify_formality(generated))

    #   score GYAFC-zero_shoot accuracy
    # df = pd.read_csv("outputs/GYAFC-zero_shoot.csv", header=None)
    # print(len(df))
    # set_trace()
    # generated = list(df.apply(remove_end_quote, axis=1))
    # score_formality(generated, target_class="formal", output_file="results/GYAFC_score.csv")

    #   score GYAFC-zero_shoot bleu
    # df = pd.read_csv("outputs/GYAFC-zero_shoot.csv", header=None)
    # generated = list(df.apply(remove_end_quote, axis=1))

    # df2 = pd.read_csv("GYAFC/GYAFC_test.csv")
    # refs = df2[['ref0', 'ref1', 'ref2', 'ref3']].transpose().values.tolist()
    # set_trace()

    # score_BLEU(generated, refs)

    #   score GYAFC-zero-shoot ppl
    df = pd.read_csv("outputs/GYAFC-zero_shoot.csv", header=None)
    generated_list = list(df.apply(remove_end_quote, axis=1))
    print(score_ppl(generated_list))
