
from transformers import BertTokenizer


def main():
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")  # pre-trained on colloquial data
    print(tokenizer.decode(410))  # 긕
    print(tokenizer.decode(411))  # 긘


if __name__ == '__main__':
    main()
