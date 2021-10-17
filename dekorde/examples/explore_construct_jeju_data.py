import json
import os
import re

import pandas as pd
from functools import reduce
from pathlib import Path
from pprint import pprint as pp


def construct_jeju_data():
    p = Path('../../data/jeju/raw')
    data_objs = map(
        lambda pth: json.load(open('/'.join(pth.parts))),
        p.iterdir()
    )

    data_objs = reduce(
        lambda i, j: i + j,
        map(
            lambda data_obj: list(
                filter(
                    lambda case: case is not None,
                    map(
                        lambda case:
                        (case['standard_form'], case['dialect_form'])
                        if case['standard_form'] != case['dialect_form']
                        else None,
                        data_obj['utterance']
                    )
                )
            ),
            data_objs
        )
    )

    df = pd.DataFrame(data_objs, columns=('seoul', 'jeju'))
    df.to_csv(os.path.join('/'.join(p.parent.parts), 'jeju.tsv'), sep='\t', index=False)


def post_process():
    def _select_word(s: str, cat: str):
        if '/' not in s:
            return s

        if cat == 'seoul':
            words = s.split('/')
            left = re.sub(r'\(.*\)', '', words[0])
            right = s.split('/')[1].replace('(', '').replace(')', '')

            return left + ' ' + right

        if cat == 'jeju':
            words = s.split('/')
            left = s.split('/')[0].replace('(', '').replace(')', '')
            right = re.sub(r'\(.*\)', '', words[1])

            return left + ' ' + right

    df = pd.read_csv('../../data/jeju/jeju.tsv', sep='\t')

    df['seoul'] = df['seoul'].apply(lambda s: re.sub(r'\{.*?\}', '', s))
    df['jeju'] = df['jeju'].apply(lambda s: re.sub(r'\{.*?\}', '', s))

    df['seoul'] = df['seoul'].apply(lambda s: re.sub(r'\&.*?\&', '', s))
    df['jeju'] = df['jeju'].apply(lambda s: re.sub(r'\&.*?\&', '', s))

    df['seoul'] = df['seoul'].apply(lambda s: re.sub(r'\(+\)+', '', s))
    df['jeju'] = df['jeju'].apply(lambda s: re.sub(r'\(+\)+', '', s))

    df['seoul'] = df['seoul'].apply(lambda s: _select_word(s, 'seoul'))
    df['jeju'] = df['jeju'].apply(lambda s: _select_word(s, 'jeju'))

    df['seoul'] = df['seoul'].apply(lambda s: s.replace('(', ''))
    df['seoul'] = df['seoul'].apply(lambda s: s.replace(')', ''))
    df['jeju'] = df['jeju'].apply(lambda s: s.replace('(', ''))
    df['jeju'] = df['jeju'].apply(lambda s: s.replace(')', ''))

    df.to_csv('../../data/jeju/jeju.tsv', sep='\t', index=False)


if __name__ == '__main__':
    post_process()
