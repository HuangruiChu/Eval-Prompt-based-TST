import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer


def classify(pred_texts):
    """
    Classify list of texts using a finetuned RoBERTa-Large (SiEBERT)
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

if __name__ == "__main__":
    pred_texts = ["ever since joes has changed hands it 's just gotten worse and worse ."]
    print(classify(pred_texts))
