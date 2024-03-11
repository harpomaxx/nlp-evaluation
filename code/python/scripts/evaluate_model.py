import argparse
import torch

from pynvml import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge 
from bert_score import score
from pprint import pprint
from tqdm import tqdm
import json
import pandas as pd
import sacrebleu
import nltk

nltk.download('wordnet') 
nltk.download('punkt') # tokenizer

def evaluate_responses(responses, references):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(responses, references, avg=True)
    # Calculate BLEU score
    bleu_scores = []
    ter_scores = []
    meteor_scores = []
    smoothie = SmoothingFunction().method4  # This is a smoothing method. There are several available.

    for response, reference in zip(responses, references):
        bleu_score = sentence_bleu([reference.split()], response.split(), smoothing_function=smoothie)
        bleu_scores.append(bleu_score)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

      # Calculate BERTScore
    P, R, F1 = score(responses, references, lang='en', verbose=True)
    avg_F1 = F1.mean().item()

    # Calculate TER
    for response, reference in zip(responses, references):
        ter = sacrebleu.metrics.TER().sentence_score(response, [reference])
        ter_scores.append(ter.score)

    avg_ter_score = sum(ter_scores) / len(ter_scores)

    # Calculate METEOR
    for response, reference in zip(responses, references):
        response_tok = nltk.word_tokenize(response)
        reference_tok = nltk.word_tokenize(reference)
        
        meteor_sc = meteor_score([reference_tok], response_tok)
        meteor_scores.append(meteor_sc)

    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)


    # Return both ROUGE and BLEU and BERT scores
    return {
        "BERT": avg_F1,
        "ROUGE": rouge_scores,
        "BLEU": avg_bleu_score,
        "TER": avg_ter_score,
        "METEOR": avg_meteor_score
    }
    return scores   

def load_jsonl(filename):
    df = pd.read_json(filename, lines=True)
    return   df['ground_truth'].tolist() , df['gpt-3.5-turbo-0125'].tolist() , df['ft:gpt-3.5-turbo-0125:personal::8xHKT1GF'].tolist()

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Evaluate answers using NLP techniques")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to the JSONL file with ground truth and translations.")
    args = parser.parse_args()
    gt,gpt,ft = load_jsonl(args.jsonl_path)

    scores = evaluate_responses(gpt, gt)
    pprint(scores)
    scores = evaluate_responses(ft, gt)
    pprint(scores)
