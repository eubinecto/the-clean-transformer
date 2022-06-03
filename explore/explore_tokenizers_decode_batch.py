import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from cleanformer.fetchers import fetch_kor2eng, fetch_config, fetch_tokenizer, fetch_transformer
from cleanformer import preprocess as P

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main():
    kor2eng = fetch_kor2eng()
    config = fetch_config()["transformer"]
    tokenizer = fetch_tokenizer(config["tokenizer"])
    transformer = fetch_transformer(config["ver"])
    test = TensorDataset(
        P.src(tokenizer, config["max_length"], kor2eng[2]),
        P.tgt_r(tokenizer, config["max_length"], kor2eng[2]),
        P.tgt(tokenizer, config["max_length"], kor2eng[2]),
    )
    test_dataloader = DataLoader(
        test,
        batch_size=10,
        shuffle=False,
        num_workers=os.cpu_count(),
    )
    for batch in tqdm(test_dataloader):
        src, tgt_r, tgt = batch
        tgt_hat = transformer.infer(src, tgt_r)  # this is the bottle-neck!
        translations = tokenizer.decode_batch(tgt_hat.tolist())
        labels = tokenizer.decode_batch(tgt.tolist())
        print(translations)
        print(labels)


if __name__ == "__main__":
    main()
