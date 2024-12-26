# flake8: noqa
from mmscan.mmscan import MMScan

print('MMScan module loaded')
try:
    from mmscan.evaluator.vg_evaluation import VisualGroundingEvaluator
except:
    pass
try:
    from mmscan.evaluator.qa_evaluation import QuestionAnsweringEvaluator
except:
    pass
try:
    from mmscan.evaluator.gpt_evaluation import GPTEvaluator
except:
    pass
