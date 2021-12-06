from transformers import BertTokenizer


def main():
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
    tokenizer.add_special_tokens({'eos_token': "[EOS]"})
    print(tokenizer.eos_token)


if __name__ == '__main__':
    main()
