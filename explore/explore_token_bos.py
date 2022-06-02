from transformers import BertTokenizer


def main():
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
    tokenizer.add_special_tokens({"bos_token": "[BOS]"})
    token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    print(token_id)
    print(tokenizer.bos_token)
    print(tokenizer.decode(1))


if __name__ == "__main__":
    main()
