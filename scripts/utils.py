import sacrebleu

def counter_dict2list(counter_dict):
    result = []
    for k, v in counter_dict.items():
        result.extend([k] * int(v))

    return result


def bleu(references, hypotheses):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    bleu_scores = sacrebleu.raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
    return scores