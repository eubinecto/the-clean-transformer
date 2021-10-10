import torch
from keras_preprocessing.text import Tokenizer

from dekorde.builders import build_I
from dekorde.loaders import load_gibberish2kor


def explore_build_I():
    gibberish2kor = load_gibberish2kor()

    gibs = [row[0] for row in gibberish2kor]
    kors = [row[1] for row in gibberish2kor]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts=gibs + kors)

    res = build_I(
        sents=['에어컨이 너무 안 시원해요.'],
        tokenizer=tokenizer,
        max_length=20,
        device=torch.device('cpu')
    )

    print(res)

    """
    tensor([[351,   1, 353, 354,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0]])
    
    '너무': 1,      
    '에어컨이': 351, 
    '별로': 352, 
    '안': 353, 
    '시원해요': 354,
    """


if __name__ == '__main__':
    explore_build_I()
