# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import json
import pdb

from .cider_scorer import CiderScorer


class Cider:
    """Main Class to compute the CIDEr metric."""

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, gts, res):
        """Main function to compute CIDEr score.

        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>  pred
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence> gt
        :return: cider (float) : computed CIDEr score for the corpus
        """

        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) >= 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return 'CIDEr'


if __name__ == '__main__':
    x = Cider()
    with open('/home/zhuziyu/work/vlpr/3dVL/scan2cap_result.json', 'r') as f:
        json_file = json.load(f)
        #print(json_file['gt_sentence_mp'])
        #print(x.compute_score({"scene_001": ["This is a chair"], "scene_002": ["That is a book"]}, {"scene_001": ["This is a chair"], "scene_002": ["That is a book"]}))
        print(
            x.compute_score(json_file['gt_sentence_mp'],
                            json_file['pred_sentence_mp']))
