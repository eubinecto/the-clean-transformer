"""
트랜스포머를 학습할 때 사용되는 입력과 레이블
"""
from cleanformer.builders import TrainInputsBuilder, LabelsBuilder
from cleanformer.fetchers import fetch_tokenizer

def main():
    tokenizer = fetch_tokenizer(entity="eubinecto", ver="wp")
    max_length = 10
    inputs_builder = TrainInputsBuilder(tokenizer, max_length)
    labels_builder = LabelsBuilder(tokenizer, max_length)

    kors = ["난 행복해"]
    engs = ["I'm on cloud nine"]

    X = inputs_builder(srcs=kors, tgts=engs)  # (N, 2(src/tgt), 2(ids/mask), L)
    Y = labels_builder(tgts=engs)  # (N, L)

    # 먼저, X의 shape을 확인하기
    print(X.size())  # (N, 2, 2, L)
    src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
    tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]
    print("--- ids ---")
    print(src_ids, src_ids.size())
    print(src_key_padding_mask, src_key_padding_mask.size())
    print(tgt_ids, src_ids.size())
    print(tgt_key_padding_mask, tgt_key_padding_mask.size())
    print(Y, Y.size())
    print("--- decoded --- #")
    print("입력 텐서 1 (인코더의 입력) - 소스 문장:")
    print([tokenizer.id_to_token(src_id) for src_id in src_ids.squeeze().tolist()])
    print("정답 텐서")
    print([tokenizer.id_to_token(y_id) for y_id in Y.squeeze().tolist()])
    print("입력 텐서 2 (디코더의 입력, 정답 텐서를 right-shift by 1 position) - 타겟 문장:")
    print([tokenizer.id_to_token(tgt_id) for tgt_id in tgt_ids.squeeze().tolist()])


if __name__ == '__main__':
    main()