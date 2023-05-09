import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import AutoModelForCausalLM
import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import argparse
import datetime

from pdb import set_trace


SENTIMENT_MAP = {
    "negative": 0,
    "positive": 1
}

FORMALITY_MAP = {
    "informal": 0,
    "formal": 1
}
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

def classify_sentiment(pred_texts):
    """Classify list of texts using a finetuned RoBERTa-Large (SiEBERT)
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

def score_sentiment(generated, target_class, output_file=None):
    """Give the fraction and percentage of successful sentiment transfers
    
    Assumes all of the generated sentences are trying to be the target_class
    """

    assert target_class in SENTIMENT_MAP

    preds, _, _ = classify_sentiment(generated)
    
    total = len(preds)
    score = sum(preds)
    if SENTIMENT_MAP[target_class] == 0:
        score = total - sum(preds)

    if output_file:
        expected_scores = total * [SENTIMENT_MAP[target_class]]
        individual_scores = [classified == expected for (classified, expected) in zip(preds, expected_scores)]
        print(f"Saving in new file {output_file}")
        pd.DataFrame(individual_scores).to_csv(output_file, index=False, header=None)

    return f"{(score)/total:.2f} ({score}/{total})"
    
def score_formality(generated, target_class="formal", output_file=None):
    """Give the fraction and percentage of successful formality transfers
    
    Assumes all of the generated sentences are trying to be the target_class

    {0: "informal", 1: "formal"}
    """

    assert target_class in FORMALITY_MAP

    preds, _, _ = classify_formality(generated)

    total = len(preds)
    score = sum(preds)
    if FORMALITY_MAP[target_class] == 0:
        score = total - sum(preds)
    
    if output_file:
        expected_scores = total * [FORMALITY_MAP[target_class]]
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

def save_score(result_path, score_str):
    print(score_str)
    if result_path:
        with open(result_path, 'a') as file:
            file.write(f'{score_str}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model output using various metrics.')
    parser.add_argument('generated_path', type=str, help='path to csv file contaning model generations')
    parser.add_argument('--refs_path', type=str, help='path to csv file containing human references', default=None)
    parser.add_argument('--remove_end_quote', action='store_true', help='look for and remove any end quotes in the generations')
    parser.add_argument('--ppl', action='store_true', help='calculate perplexity score')
    parser.add_argument('--bleu', action='store_true', help='calculate bleu score')
    parser.add_argument('--formality', type=str, default=None, choices=FORMALITY_MAP, help='calculate formality score')
    parser.add_argument('--sentiment', type=str, default=None, choices=SENTIMENT_MAP, help='calculate sentiment score')
    parser.add_argument('--result_path', type=str, default=None, help='path to append the score results')

    args = parser.parse_args()

    generated_path = args.generated_path
    refs_path = args.refs_path
    remove_end_quote = args.remove_end_quote
    find_ppl = args.ppl
    find_bleu = args.bleu
    find_formality = args.formality
    find_sentiment = args.sentiment
    result_path = args.result_path

    df = pd.read_csv(generated_path, header=None)
    if remove_end_quote:
        generated = list(df.apply(remove_end_quote, axis=1))
    else:
        generated = list(df[0])

    # calculate scores...
    save_score(result_path, f"{datetime.datetime.now().time()}")
    if find_ppl:
        print("Calculating PPL...")
        ppl_score = f"PPL: {score_ppl(generated)}"
        save_score(result_path, ppl_score)
    if find_bleu:
        print("Calculating BLEU...")
        if not refs_path:
            bleu_score = "BLEU: N/A (No refs path provided)"
        else:
            df_refs = pd.read_csv(refs_path)
            refs = df_refs[['ref0', 'ref1', 'ref2', 'ref3']].transpose().values.tolist()
            bleu_score = f"BLEU: {score_BLEU(generated, refs)}"
        save_score(result_path, bleu_score)

    if find_formality:
        print(f"Calculating formality acc for target class {find_formality}...")
        formality_acc = f"Formality acc: {score_formality(generated, target_class=find_formality)}"
        save_score(result_path, formality_acc)
    if find_sentiment:
        print(f"Calculating sentiment acc for target class {find_formality}...")
        sentiment_acc = f"Sentiment acc: {score_sentiment(generated, target_class=find_sentiment)}"
        save_score(result_path, formality_acc)

