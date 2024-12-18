import json
import random
import threading
from typing import List

from openai import OpenAI
from tqdm import tqdm

from mmscan.utils.lang_utils import qa_metric_map, qa_prompt_define


class GPTEvaluator:
    """GPT metric, we set this for QA and Caption tasks.

    Args:
        eval_size (int) : The number of samples to evaluate, -1 means
            all samples.
            Defaults to -1.
        api_key (str) : The openai key.
        model (str) : The GPT model to use, default we use "gpt-4o-mini".
            Defaults to "gpt-4o-mini".
        show_progress (bool) : Whether to print the evaluation results or not.
            Defaults to False.
    """

    def __init__(self,
                 eval_size: int = -1,
                 api_key: str = '',
                 model: str = 'gpt-4o-mini',
                 show_progress: bool = False):
        self.eval_size = eval_size
        self.model = model
        self.show_progress = show_progress
        self.client = OpenAI(api_key)
        self.qa_metric = [
            'STa',
            'STs',
            'OOa',
            'OOs',
            'OR',
            'overall',
            'Advanced',
        ]

    def normal_query(self,
                     system_prompt: str,
                     user_content_groups: List[str],
                     max_tokens: int = 1000) -> dict:
        """Calling the GPT api, return the results in the format of json.

        Args:
            system_prompt (str) :
                The system prompt inputted into GPT.
            user_content_grounps (list[str]) :
                The user content inputted into GPT.
            max_tokens (int) : Max tokens. Defaults to 1000.

        Returns:
            dict : The json-format result.
        """

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        for content_group in user_content_groups:
            messages.append({'role': 'user', 'content': content_group})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={'type': 'json_object'},
            max_tokens=max_tokens,
        )
        response = json.loads(response.choices[0].message.content)
        return response

    def qa_evaluation(self, all_samples: dict, thread_index: int,
                      tmp_path: str) -> None:
        """Employ the GPT evaluator.

        Args:
            all_samples (dict) : The QA sample dict with QA_ID as keys and
                [gt, pred, question] as values.
            thread_index (int) : The index of the thread.
            tmp_path (str) : The path to store the
                tmp-stored json files.
        """

        system_prompt, ex_instance = qa_prompt_define()

        # Define the number of retries
        MAXTRY = 3
        gpt_eval_results = {}

        for sample_id in tqdm(all_samples):
            GPT_INTPUT = {
                'Question': all_samples[sample_id]['question'],
                'Model Answer': all_samples[sample_id]['pred'],
                'Human Answer': all_samples[sample_id]['gt'][0],
            }

            for _ in range(MAXTRY):
                FLAG = False
                try:
                    GPT_OUTPUT = self.normal_query(system_prompt + ex_instance,
                                                   [str(GPT_INTPUT)])
                    # check the result forms
                    assert ('All key points' in GPT_OUTPUT
                            and 'Correct Number' in GPT_OUTPUT
                            and 'Wrong/Missing Number' in GPT_OUTPUT
                            and 'Reasons' in GPT_OUTPUT)
                    assert (len(GPT_OUTPUT['All key points'])
                            == int(GPT_OUTPUT['Correct Number']) +
                            int(GPT_OUTPUT['Wrong/Missing Number'])
                            and len(GPT_OUTPUT['All key points']) > 0)

                    FLAG = True
                except Exception:

                    continue
                if FLAG:
                    gpt_eval_results[sample_id] = GPT_OUTPUT

        with open(
                tmp_path.replace('.json',
                                 '_thread' + str(thread_index) + '.json'),
                'w',
        ) as f:
            json.dump(gpt_eval_results, f, indent=4)

    def qa_collection(self, num_threads: int, tmp_path: str) -> dict:
        """Collect the gpt-eval results from the tmp-stored json files.

        Args:
            num_threads (int) :
                The number of threads used to evaluate the samples.
            tmp_path (str) :
                The path to store the tmp-stored json files.

        Returns:
            dict : The evaluation result.
        """

        eval_dict = {metric: [] for metric in self.qa_metric}
        static_result = {}
        for thread_index in range(num_threads):
            with open(
                    tmp_path.replace('.json',
                                     '_thread' + str(thread_index) + '.json'),
                    'r',
            ) as f:
                thread_result = json.load(f)
            for qa_id in thread_result:
                static_result[qa_id] = thread_result[qa_id]
        for qa_id in static_result:
            if len(static_result[qa_id]['All key points']) == 0:
                continue
            eval_dict[qa_metric_map(qa_id.split('__')[0])].append(
                int(static_result[qa_id]['Correct Number']) /
                (int(static_result[qa_id]['Correct Number']) +
                 int(static_result[qa_id]['Wrong/Missing Number'])))
            eval_dict['overall'].append(
                int(static_result[qa_id]['Correct Number']) /
                (int(static_result[qa_id]['Correct Number']) +
                 int(static_result[qa_id]['Wrong/Missing Number'])))
        for metric in eval_dict:
            eval_dict[metric] = (sum(eval_dict[metric]) /
                                 len(eval_dict[metric])
                                 if len(eval_dict[metric]) > 0 else None)

        return eval_dict

    def load_and_eval(self,
                      raw_batch_input: List[dict],
                      num_threads: int = 1,
                      tmp_path: str = './') -> dict:
        """Load the batch of results and evaluate.

        Args:
            raw_batch_input (list[dict]) :
                The batch of results wanted to evaluate
            num_threads (int) : The number of the threadings.
                Defaults to 1.
            tmp_path (str) : The temporay path to store the json files.

        Returns:
            dict : The evaluation result.
        """

        # (1) Update the results and store in the dict.

        batch_result = {}
        self.__check_format__(raw_batch_input)
        for _input in raw_batch_input:
            batch_result[_input['ID']] = _input

        # (2) Evaluate the QA task.
        if self.eval_size == -1:
            num_sample = len(batch_result)
        else:
            num_sample = self.eval_size
        qa_sample = random.sample(list(batch_result.keys()), num_sample)
        threads = []
        qa_ids = list(qa_sample)
        IDs_divide_index = []
        for _index in range(num_threads):
            IDs_divide_index.append(
                qa_ids[len(qa_ids) // num_threads * _index:len(qa_ids) //
                       num_threads * (_index + 1)])

        for thread_index in range(num_threads):
            # Create a sub-dictionary for each thread
            partial_samples = {
                ID_: batch_result[ID_]
                for ID_ in IDs_divide_index[thread_index]
            }
            if self.show_progress:
                print(
                    f'Thread {thread_index} processing {len(partial_samples)}')
            thread = threading.Thread(
                target=self.qa_evaluation,
                args=(partial_samples, thread_index,
                      tmp_path + '/gpt_QA.json'),
            )
            threads.append(thread)

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if self.show_progress:
            print(f'the results are store under {tmp_path}')

        # (3) Collect the results.
        eval_dict = self.qa_collection(num_threads, tmp_path + '/gpt_QA.json')

        return eval_dict

    def __check_format__(self, raw_input):
        """Check if the input conform with mmscan evaluation format. The input
        to be checked, should be a list of dict. Every item with the keys:

        ["ID","question","pred",""gt"] pred is a list with one one element. gt
        is a list with >=1 elements. "ID" should be unique.

        Args:
            raw_input (list[dict]) : The input to be checked.
        """
        assert isinstance(
            raw_input,
            list), 'The input of MMScan evaluator should be a list of dict. '

        for _index in range(len(raw_input)):
            assert 'ID' in raw_input[_index]
            assert ('pred' in raw_input[_index]
                    and isinstance(raw_input[_index]['pred'], list)
                    and len(raw_input[_index]['pred']) == 1)
            assert ('gt' in raw_input[_index]
                    and isinstance(raw_input[_index]['gt'], list)
                    and len(raw_input[_index]['gt']) >= 1)
            assert 'question' in raw_input[_index]
