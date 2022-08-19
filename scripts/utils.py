import sacrebleu

def counter_dict2list(counter_dict):
    result = []
    for k, v in counter_dict.items():
        result.extend([k] * int(v))

    return result


def bleu(predictions, references):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    bleu_scores = sacrebleu.raw_corpus_bleu(predictions, [references]).precisions
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = round(bleu_scores[n], 2)
    return scores