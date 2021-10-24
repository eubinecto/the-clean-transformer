from typing import List
from transformers import BertTokenizer
from dekorde.components.transformer import Transformer


class Dekorder:
    def __init__(self, transformer: Transformer, tokenizer: BertTokenizer):
        self.transformer = transformer
        self.tokenizer = tokenizer

    def __call__(self, jejus: List[str]):
        """
        제주도 방언 -> dekorde -> 서울말
        :param jejus:
        :return seouls:
        """
        # TODO
        pass
