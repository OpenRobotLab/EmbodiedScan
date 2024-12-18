# flake8: noqa
# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/data/data_utils.py#L322-L379
import re
import string


def clean_answer(data):
    """Help to clean and unify the sentence.

    Args:
        data (str): the raw sentence.

    Returns:
        data (str): the processed sentence.
    """

    data = data.lower()
    data = re.sub('[ ]+$', '', data)
    data = re.sub('^[ ]+', '', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub("[^a-zA-Z0-9,'\s\-:]+", '', data)
    data = re.sub('ç', 'c', data)
    data = re.sub('’', "'", data)
    data = re.sub(r'\bletf\b', 'left', data)
    data = re.sub(r'\blet\b', 'left', data)
    data = re.sub(r'\btehre\b', 'there', data)
    data = re.sub(r'\brigth\b', 'right', data)
    data = re.sub(r'\brght\b', 'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b', 'TV', data)
    data = re.sub(r'\bchai\b', 'chair', data)
    data = re.sub(r'\bwasing\b', 'washing', data)
    data = re.sub(r'\bwaslked\b', 'walked', data)
    data = re.sub(r'\boclock\b', "o'clock", data)
    data = re.sub(r"\bo\'[ ]+clock\b", "o'clock", data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b', r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


def normalize_answer(s):
    """Help to 'normalize' the answer.

    Args:
        s (str): the raw answer.

    Returns:
        str : the processed sentence.
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """Collect the refined exact match score between prediction and ground
    truth.

    Args:
        prediction (str): thr predicted answer.
        ground_truth (str): the gt answer.

    Returns:
        float : the exact match score
    """

    return normalize_answer(prediction) == normalize_answer(ground_truth)


def special_token_filter(lan, clean=True, truncation=True, max_length=1024):
    """
    Usage:
        clean the language, remove stop words and special tokens
    Args:
        lan: List[str], language to be cleaned
        clean: bool, if apply LEO clean strategy
        truncation: to avoid crash pycocoevalcap the
            input sentence will be truncated to max_length
        max_length: You may set this to the max length of possible gt answer
    """

    replacements = {
        'ASSISTANT:': '',
        'ASSISTANT: ': '',
        '\n': '',
        '<s>': '',
        '</s>': '',
        '<unk>': '',
        '<p>': '',
        '</p>': '',
        '<ref>': '',
        '<|endoftext|>': '',  # for GPT2
    }
    for old, new in replacements.items():
        lan = lan.replace(old, new)
    lan = lan.strip()
    lan = re.sub(r'\s{2,}', ' ', lan)
    if truncation:
        if len(lan) > max_length:
            lan = lan[:max_length]
    if clean:
        lan = clean_answer(lan)
    return lan


def qa_prompt_define():
    """Define the system prompt and example instance.

    Returns:
        system_prompt : str, system prompt input into GPT
        ex_instance : str, example instance of the input and expected output in JSON format
    """

    system_prompt = (
        'Evaluate a model-generated QA result against a human-generated answer for a 3D model.'
        +
        ' I will give you a dict with "Question", "Model Answer" and "Human Answer".Please fully understand the'
        +
        ' meaning of both s and  follow these three steps to evaluate the model-generated answer: First step, '
        +
        'identify all key points in the human answer and list them; Scecond step, compare each of these key points'
        +
        ' in the model-generated answer, count the number of key points which are correct in the model-generated '
        +
        'answer, also, count the number of key points which are missing or error in the model-generated answer. '
        +
        'Provide reasons for each evaluation, you should not be too strict, as long as a key point has no significant'
        +
        ' difference in model and human, regard it as correct one; Third step, output the "All key points" (list), '
        +
        '"Correct Number" (int), "Wrong/Missing Number"(int), "Reasons" (str) in a JSON format.( Obviously the '
        +
        '"Correct Number"+"Wrong/Missing Number" should be equal to the total number of key points). Give you some examples: '
    )
    ex_instance = (
        'The input is: { "Question" : "What is the function of this object?", "Model Answer" : '
        +
        '"It can hang clothes for storage.", "Human Answer" : "Providing storage space." }, the expected '
        +
        'output is { "All key points" : ["function of the object: providing storage space"], "Correct '
        +
        'Number" : 1, "Wrong/Missing Number" : 0, "Reasons" : "A place for hanging clothes also provides storage space." }; '
    )
    ex_instance += (
        'The input is: { "Question" : "What is the placement of this object?", "Model Answer" : "It is '
        +
        'placed vertically on the table", "Human Answer" : "It placement is standing upright on the floor. " }, the '
        +
        'expected output is { "All key points" : ["placement of the object: standing upright","surface the object is '
        +
        'standing upright on: on the floor"], "Correct Number" : 1, "Wrong/Missing Number" : 1, "Reasons" : "The model'
        +
        ' correctly identifies the object is standing but fails to identify the surface it is standing on." }; '
    )
    ex_instance += (
        'The input is { "Question" : "Please compare these two objects closely, are they similar in '
        +
        'material? Given me the answer and reason.", "Model Answer" : "No, because the object is made of plastic '
        +
        'and the pillow is made of cotton.", "Human Answer" : "No,  because the bowl is wooden and the pillow is '
        +
        'soft fabric."}, the expected output is { "All key points" : ["Yes or No : No","Texture of the bowl : '
        +
        'wooden ","Texture of the pillow : soft fabric"], "Correct Number" : 2, "Wrong/Missing Number" : 1,'
        +
        ' "Reasons" : "The model correctly identifies the material of pillow (cotton is soft fabric) but '
        + 'fails to recognize the material of the bowl." }. ')

    return system_prompt, ex_instance


def qa_metric_map(eval_type):
    """Map the class type to the corresponding Abbrev.

    Args:
        eval_type (str): the class name.

    Returns:
        str : the corresponding Abbrev.
    """
    if 'Attribute_OO' in eval_type:
        target = 'OOa'
    elif 'Space_OO' in eval_type:
        target = 'OOs'
    elif 'EQ' in eval_type or 'Single_Attribute' in eval_type:
        target = 'STa'
    elif 'OR' in eval_type:
        target = 'OR'
    elif 'Single_Space' in eval_type:
        target = 'STs'
    elif 'Advanced' in eval_type:
        target = 'Advanced'
    return target
