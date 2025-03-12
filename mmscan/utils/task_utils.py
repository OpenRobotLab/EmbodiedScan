from typing import List


def anno_token_flatten(samples: List[dict], keep_only_one: bool = True):
    """Flatten the annotation tokens for each target in a 3d visual grounding
    sample.

    Args:
        samples (list[dict]): The original VG samples.
        keep_only_one (bool):
            Whether to keep only one positive token for each target.
            Defaults to True.

    Returns:
        List[dict] : The token-flattened samples.
    """

    marked_indices = []
    for i, d in enumerate(samples):
        if 'target_id' not in d:
            continue
        target_ids = d['target_id']
        ret_target_ids = []
        ret_target = []
        ret_tps = []
        for i, target_id in enumerate(target_ids):
            tps = d['tokens_positive'].get(str(target_id), [])
            for tp in tps:
                ret_target_ids.append(target_id)
                ret_target.append(d['target'][i])
                ret_tps.append(tp)
                if keep_only_one:
                    break
        d['target_id'] = ret_target_ids
        d['target'] = ret_target
        d['tokens_positive'] = ret_tps
        if len(d['target_id']) == 0:
            marked_indices.append(i)

    for i in marked_indices[::-1]:
        del samples[i]

    return samples
