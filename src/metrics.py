import yaml
import numpy as np
from evaluate import load
from typing import List


bleu = load('bleu')
sacrebleu = load("sacrebleu")
meteor = load('meteor')
chrf = load('chrf')
bertscore = load('bertscore')
bleurt = load('bleurt', module_type='metric')
comet = load('comet')


def evaluate_metrics(predictions: List[str], references: List[str], sources: List[str]) -> None: 
    references_lists = [[text] for text in references]

    P, R, F1, _ = bertscore.compute(predictions=predictions, references=references, lang='pl').values()
    avg_precision = np.mean(P).item()
    avg_recall = np.mean(R).item()
    avg_f1 = np.mean(F1).item()

    metrics = {
        'BLEU': bleu.compute(predictions=predictions, references=references_lists)['bleu'],
        'SCAREBLEU': sacrebleu.compute(predictions=predictions, references=references_lists)['score'],
        'METEOR': meteor.compute(predictions=predictions, references=references_lists)['meteor'].item(),
        'chrF': chrf.compute(predictions=predictions, references=references_lists)['score'],
        'BERTScore': {
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1
        },
        'BLEURT': np.mean(bleurt.compute(predictions=predictions, references=references)['scores']).item(),
        'COMET': comet.compute(predictions=predictions, references=references, sources=sources)['mean_score']
    }

    with open('metrics.yaml', 'w') as file:
        yaml.dump(metrics, file)
