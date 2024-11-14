from mmscan.mmscan import MMScan

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
