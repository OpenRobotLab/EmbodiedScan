from collections import defaultdict
from typing import List, Tuple

import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer


def to_coco(kvs, keys):
    res = defaultdict(list)
    for k in keys:
        if k in kvs:
            caps = kvs[k]
            for c in caps:
                res[k].append({'caption': c})
        else:
            res[k].append({'caption': ''})
    return res


def coco_evaluation(batch_input: List[dict]) -> Tuple[dict, dict]:
    """Calculate the extract matching score for each item.
    Args:
        batch_input(list[dict]):
            [{
                "pred": [str],
                "gt":[str,...]
            },...]

    Returns:
        dict, dict: final_scores stores the score of each metric
    """

    prediction = {}
    ground_truths = {}

    for _input in batch_input:
        prediction[_input['ID']] = _input['pred']
        ground_truths[_input['ID']] = _input['gt']

    scorers = [
        (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
        (Meteor(), 'METEOR'),
        (Rouge(), 'ROUGE_L'),
        (Cider(), 'CIDEr'),
        (Spice(), 'SPICE'),
    ]

    tokenizer = PTBTokenizer()
    ref_sent = ground_truths
    hypo_sent = prediction
    final_scores = {}
    final_list = {}
    ref_coco = tokenizer.tokenize(to_coco(ref_sent, ref_sent.keys()))
    hypo_coco = tokenizer.tokenize(to_coco(hypo_sent, ref_sent.keys()))

    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref_coco, hypo_coco)
        if type(score) == list:
            for m, s, s_ in zip(method, score, scores):
                final_scores[m] = s
                final_list[m] = s_
        else:
            final_scores[method] = score
            final_list[method] = scores

    return final_scores, final_list


def em_evaluation(batch_input: List[dict]) -> Tuple[list, list]:
    """Calculate the extract matching score for each item.
    Args:
        batch_input(list[dict]):
            [{
                "pred": [str],
                "gt":[str,...]
            },...]

    Returns:
        list[float]: (refined) extract matching score for each item
    """
    # EM
    em_result = []
    for _input in batch_input:
        pred = _input['pred'][0]
        gts = _input['gt']
        if pred in gts:
            em_result.append(1)
        else:
            em_result.append(0)

    # refined EM
    refine_em_result = []

    for _input in batch_input:
        correct = 0
        pred = _input['pred'][0]
        gts = _input['gt']

        if len(pred.split()) == 0:
            pred = '@@@@@@@@-= Empty Answer =-@@@@@@@@@'
        for gt in gts:
            if pred == gt:
                correct = 1
                continue
            elif ''.join(pred.split()) in ''.join(gt.split()):
                correct = 1
                continue
            elif ''.join(gt.split()) in ''.join(pred.split()):
                correct = 1
                continue
        refine_em_result.append(correct)
    return em_result, refine_em_result


class SimCSEEvaluator:
    """A class for calculating the simcse similarity score. Using Sentence
    Embeddings to calculate similarity between pred/gt。

    Args:
        model_path: path to the simcse pretrained model.
    """

    def __init__(self, model_path: str, eval_bs: int = 500) -> None:
        if len(model_path) == 0:
            model_path = 'princeton-nlp/sup-simcse-roberta-large'
        self.eval_bs = eval_bs
        self.simcse_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.simcse_model = AutoModel.from_pretrained(model_path).to('cuda')

    def __batch_evaluation__(self, all_pred: List[str], all_gt: List[str],
                             gt_count: List[int]) -> List[float]:
        """Using Sentence Embeddings to calculate similarity between pred/gt in
        a batch.

        Args:
            all_pred(list[str]): all prediction
            all_gt(list[str]): all ground truth
            gt_count(list[int]):
                stores number of possible answers to a question
            tips: len(all_gt)>=len(all_pred)
                there may be multiple gt answers for a question.

        Return:
            list[float]: Simcse similarity of each pred/gts pair.
        """
        len_of_pred = len(all_pred)
        with torch.no_grad():
            inputs = self.simcse_tokenizer(
                all_pred + all_gt,
                padding=True,
                truncation=True,
                return_tensors='pt',
            ).to('cuda')
            simcse_embeddings = self.simcse_model(
                **inputs, output_hidden_states=True,
                return_dict=True).pooler_output

        all_pred_simcse_embed = simcse_embeddings[:len_of_pred]
        all_gt_simcse_embed = simcse_embeddings[len_of_pred:]
        all_simcse_sim = []

        accumulated = 0
        for i in range(len(all_pred)):
            simcse_similarity = -100
            for j in range(accumulated, accumulated + gt_count[i]):
                simcse_similarity = max(
                    simcse_similarity,
                    1 - cosine(
                        all_pred_simcse_embed[i].cpu().detach().numpy(),
                        all_gt_simcse_embed[j].cpu().detach().numpy(),
                    ),
                )

            all_simcse_sim.append(simcse_similarity)
            accumulated += gt_count[i]
        torch.cuda.empty_cache()
        return all_simcse_sim

    def evaluation(self, batch_input: List[dict]) -> List[float]:
        """Calculate the simcse similarity score for each item.
        Args:
            batch_input(list[dict]):
                [{
                    "pred": [str],
                    "gt":[str,...]
                },...]

        Returns:
            list[float]: simcse similarity for each item
        """
        all_simcse_similarity = []
        batch_lan_pred = []
        batch_lan_gt = []
        count_gt = []

        for idx, _item in enumerate(batch_input):
            batch_lan_pred.extend(_item['pred'])
            batch_lan_gt.extend(_item['gt'])
            count_gt.extend([len(_item['gt'])])

            if (idx + 1) % self.eval_bs == 0 or idx == len(batch_input) - 1:
                all_simcse_similarity += self.__batch_evaluation__(
                    batch_lan_pred, batch_lan_gt, count_gt)
                batch_lan_pred = []
                batch_lan_gt = []
                count_gt = []

        return all_simcse_similarity


class SBERTEvaluator:
    """A class for calculating the sbert similarity score. Using Sentence-BERT
    to calculate similarity between pred/gt.

    Args:
        model_path: path to the sbert pretrained model.
    """

    def __init__(self, model_path: str, eval_bs: int = 500) -> None:
        if len(model_path) == 0:
            model_path = 'all-mpnet-base-v2'
        self.eval_bs = eval_bs
        self.sbert_model = SentenceTransformer(model_path, device='cuda')

    def __batch_evaluation__(self, all_pred: List[str], all_gt: List[str],
                             gt_count: List[int]) -> List[float]:
        """Using Sentence-BERT to calculate similarity between pred/gt in a
        batch.

        Args:
            all_pred(list[str]): all prediction
            all_gt(list[str]): all ground truth
            gt_count(list[int]): stores number of possible
                answers to a question
            tips: len(all_gt)>=len(all_pred) because there may be multiple
                  gt answers for a question.

        Return:
            list[float]: Sentence-BERT similarity of each pred/gts pair.
        """
        len_of_pred = len(all_pred)
        with torch.no_grad():
            sbert_embeddings = self.sbert_model.encode(all_pred + all_gt,
                                                       show_progress_bar=False,
                                                       device='cuda')

        all_pred_sbert_embed = sbert_embeddings[:len_of_pred]
        all_gt_sbert_embed = sbert_embeddings[len_of_pred:]
        all_sbert_sim = []

        accumulated = 0
        for i in range(len(all_pred)):
            sbert_similarity = -100
            for j in range(accumulated, accumulated + gt_count[i]):
                sbert_similarity = max(
                    sbert_similarity,
                    util.cos_sim(all_pred_sbert_embed[i],
                                 all_gt_sbert_embed[j])[0][0].item(),
                )
            all_sbert_sim.append(sbert_similarity)
            accumulated += gt_count[i]
        torch.cuda.empty_cache()
        return all_sbert_sim

    def evaluation(self, batch_input: List[dict]) -> List[float]:
        """Calculate the simcse similarity score for each item.
        Args:
            batch_input(list[dict]):
                [{
                    "pred": [str],
                    "gt":[str,...]
                },...]

        Returns:
            list[float]: simcse similarity for each item
        """
        all_sbert_similarity = []
        batch_lan_pred = []
        batch_lan_gt = []
        count_gt = []

        for idx, _item in enumerate(batch_input):
            batch_lan_pred.extend(_item['pred'])
            batch_lan_gt.extend(_item['gt'])
            count_gt.extend([len(_item['gt'])])

            if (idx + 1) % self.eval_bs == 0 or idx == len(batch_input) - 1:
                all_sbert_similarity += self.__batch_evaluation__(
                    batch_lan_pred, batch_lan_gt, count_gt)
                batch_lan_pred = []
                batch_lan_gt = []
                count_gt = []

        return all_sbert_similarity
