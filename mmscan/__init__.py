# flake8: noqa
from mmscan.mmscan import MMScan

print('MMScan module loaded')
try:
    from mmscan.evaluator.vg_evaluation import VG_Evaluator
except:
    pass
try:
    from mmscan.evaluator.qa_evaluation import QA_Evaluator
except:
    pass
try:
    from mmscan.evaluator.gpt_evaluation import GPT_Evaluator
except:
    pass
