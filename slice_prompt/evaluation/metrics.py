"""Closed accuracy and standard open-answer text metrics."""
import re


def normalize(text):
    return re.findall(r'\w+', text.lower())


def evaluate_records(records):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score.rouge_scorer import RougeScorer
    labeled = [row for row in records if row.get('answer') is not None]
    closed = [row for row in labeled if row['closed']]
    opened = [row for row in labeled if not row['closed']]
    result = dict(closed_accuracy=sum(normalize(row['prediction']) == normalize(row['answer']) for row in closed) / len(closed) if closed else None)
    if opened:
        references = [[normalize(row['answer'])] for row in opened]
        predictions = [normalize(row['prediction']) for row in opened]
        scorer = RougeScorer(['rougeL'], use_stemmer=True)
        result['bleu4'] = corpus_bleu(references, predictions, smoothing_function=SmoothingFunction().method1)
        result['rougeL'] = sum(scorer.score(row['answer'], row['prediction'])['rougeL'].fmeasure for row in opened) / len(opened)
        try:
            result['meteor'] = sum(meteor_score(reference, prediction) for reference, prediction in zip(references, predictions)) / len(opened)
        except LookupError as error:
            raise RuntimeError('Install the METEOR lexical resources with: python -m nltk.downloader wordnet omw-1.4') from error
    return result
