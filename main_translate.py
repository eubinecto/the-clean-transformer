"""
Just a simple demonstration of translations
"""
from cleanformer import preprocess as P  # noqa
from cleanformer.fetchers import fetch_config, fetch_transformer, fetch_tokenizer
# --- fetch --- #
config = fetch_config()["recommended"]
transformer = fetch_transformer(config["transformer"]).eval()
tokenizer = fetch_tokenizer(config["tokenizer"])
transformer.hparams["eos_token_id"] = 3
# --- preprocess --- #
sentences = ["상황이 심각하다", "좋은 징조로 예상된다", "안녕하세요?"]
x2y = [(sent, "") for sent in sentences]
src = P.to_src(tokenizer, transformer.hparams["max_length"], x2y)
tgt_r = P.to_tgt_r(tokenizer, transformer.hparams["max_length"], x2y)
# --- translate ---- #
tgt, _ = transformer.decode(src, tgt_r)
kors = tokenizer.decode_batch(src[:, 0].tolist(), skip_special_tokens=False)
engs = tokenizer.decode_batch(tgt[:, 0].tolist(), skip_special_tokens=False)
for res in list(zip(kors, engs)):
    print(res[0])
    print(res[1])
