# the model has the ability to align text with the box

TASK_PROPMT = {
    'embodied_qa': {
        'without_box':
        dict(
            instruction=
            '### human: given the 3D scene, answer the question: "{question}" according to the given 3D scene. ### assistant:',
            answer='{answer}',
            do_localize=False),
        'with_box':
        dict(
            instruction=
            '### human: given the 3D scene, answer the question: "{question}" at "{locations}". ### assistant:',
            answer='{answer}',
            do_localize=False),
    },
    'embodied_cap': [
        dict(
            instruction=
            '### human: given the 3D scene, answer the question: "{question}" at "{locations}" ### assistant:',
            answer='{answer}',
            do_localize=False),
    ]
}
BOX_FORMAT = '<obj>{}, {}, {}, {}, {}, {}</obj>'
COORD_FORMAT = '<loc>{}, {}</loc>'
