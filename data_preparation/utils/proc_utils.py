from functools import wraps


def mmengine_track_func(func):

    @wraps(func)
    def wrapped_func(args):
        result = func(*args)
        return result

    return wrapped_func
