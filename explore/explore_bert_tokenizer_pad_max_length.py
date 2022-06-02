from transformers import BertTokenizer


sents = ["안녕", "안녕하세요"]


def main():
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
    encoded = tokenizer(
        sents,
        truncation=True,
        # you can choose your padding strategy here.
        # https://discuss.huggingface.co/t/how-to-pad-tokens-to-a-fixed-length-on-a-single-sentence/6248/2
        padding="max_length",
        max_length=40,
        return_tensors="pt",
    )
    print(tokenizer.all_special_tokens)
    print(tokenizer.bos_token)
    print(tokenizer.decode(84))
    print(encoded["input_ids"].shape)


if __name__ == "__main__":
    main()
