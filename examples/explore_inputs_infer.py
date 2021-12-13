"""
사전학습된 트랜스포머로 번역을 시도할 때 사용되는 입력.
학습을 진행하는 것이 아니므로, 레이블은 없다.
"""
from cleanformer.builders import InferInputsBuilder
from cleanformer.fetchers import fetch_tokenizer

tokenizer = fetch_tokenizer(entity="eubinecto", ver="wp")
max_length = 10
inputs_builder = InferInputsBuilder(tokenizer, max_length)

kors = ["난 행복해"]

X = inputs_builder(srcs=kors)  # (N, 2(src/tgt), 2(ids/mask), L)

# 먼저, X의 shape 확인하기
print(X.size())  # (N, 2, 2, L)
src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]
print("--- ids ---")
print(src_ids, src_ids.size())
print(src_key_padding_mask, src_key_padding_mask.size())
print(tgt_ids, src_ids.size())
print(tgt_key_padding_mask, tgt_key_padding_mask.size())
print("--- decoded --- #")
print("입력 텐서 1 (인코더의 입력) - 소스 문장:")
print([tokenizer.id_to_token(src_id) for src_id in src_ids.squeeze().tolist()])
print("입력 텐서 2 (디코더의 입력, 정답 텐서를 right-shift by 1 position) - 타겟 문장:")
print([tokenizer.id_to_token(tgt_id) for tgt_id in tgt_ids.squeeze().tolist()])
