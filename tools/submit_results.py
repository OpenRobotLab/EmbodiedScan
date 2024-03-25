# Copyright (c) OpenRobotLab. All rights reserved.
import mmengine

# Please modify the following content to submit your results
results_file = './test_results_mini.json'
submit_file = './submission_mini.pkl'

method = 'Baseline'
team = 'EmbodiedScan'
authors = 'EmbodiedScan Team'
email = 'taiwang.me@gmail.com'
institution = 'Shanghai AI Laboratory'
country = 'China'

# submission prototype:
# dict {
#     'method':   <str> -- name of the method
#     'team':     <str> -- name of the team, identical to the Google Form
#     'authors':                <list> -- list of str, authors
#     'e-mail':                 <str> -- e-mail address
#     'institution / company':  <str> -- institution or company
#     'country / region':       <str> -- country or region
#     'results': {
#         [identifier]:         <frame_token> -- identifier of the frame
#             dict or list, a single frame prediction
#         ,
#         ...
#     }
# }
results = mmengine.load(results_file)
submit_data = {
    'method': method,
    'team': team,
    'authors': authors,
    'e-mail': email,
    'institution': institution,
    'country': country,
    'results': results
}
mmengine.dump(submit_data, submit_file)
