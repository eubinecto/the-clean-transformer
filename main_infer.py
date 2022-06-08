"""
Just a simple script for demonstrating translation
"""
import argparse
from cleanformer import preprocess as P  # noqa
from cleanformer.translator import Translator

sents = ["상황이 심각하다", "좋은 징조로 예상된다", "안녕하세요?"]
parser = argparse.ArgumentParser()
args = parser.parse_args()
translator = Translator()
translator.transformer.hparams["eos_token_id"] = 3
kors, engs = translator(sents)
for res in list(zip(kors, engs)):
    print(res[0])
    print(res[1])
