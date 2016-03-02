import functools
import re

def compose(*functions):
    ''' evaluates functions from right to left.

        >>> add = lambda x, y: x + y
        >>> add3 = lambda x: x + 3
        >>> divide2 = lambda x: x/2
        >>> subtract4 = lambda x: x - 4
        >>> subtract1 = compose(add3, subtract4)
        >>> subtract1(1)
        0
        >>> compose(subtract1, add3)(4)
        6
        >>> compose(int, add3, add3, divide2)(4)
        8
        >>> compose(int, divide2, add3, add3)(4)
        5
        >>> compose(int, divide2, compose(add3, add3), add)(7, 3)
        8
    '''
    def inner(func1, func2):
        return lambda *x, **y: func1(func2(*x, **y))
    return functools.reduce(inner, functions)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub("  ", " ", string)
    return string.strip().lower()
