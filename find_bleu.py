import pandas as pd
from sacrebleu.metrics import BLEU
from pdb import set_trace

def calc_bleu(generated_list, refs_list):
    bleu = BLEU()

    # individual_scores = []
    # for generated, refs in zip(generated_list, refs_list):
    
    bleu_score = bleu.corpus_score(generated_list, refs_list)

    return bleu_score.score

df_generated = pd.read_csv("/Users/davidpeng/Desktop/some_code/Eval-Prompt-based-TST/outputs/GYAFC/GPT-paraphrase/GYAFC-GPT-paraphrase.csv")
df_refs = pd.read_csv("/Users/davidpeng/Desktop/some_code/Eval-Prompt-based-TST/GYAFC/GYAFC_test.csv")
generated_lists = list(df_generated.input)
refs_lists = df_refs[['ref0', 'ref1', 'ref2', 'ref3']].values.tolist()

bleus = [(generated_list, refs_list, calc_bleu([generated_list], [refs_list])) for generated_list, refs_list in zip(generated_lists, refs_lists)]
print([val for val in bleus if val[2] == 0])