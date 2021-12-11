# Enkorde

한-영 번역을 위한 트랜스포머 밑바닥부터 구현 (광주인공지능사관학교/멋쟁이사자차럼 자연어처리 보충수업)

## 강의노트
- [WEEK7: Transformer - why?](https://www.notion.so/WEEK7-Transformer-why-8e3712fb674a4ba2a85bf6da9cd36af0)
- [WEEK8: Transformer: How does Self-attention work?](https://www.notion.so/WEEK8-Transformer-How-does-Self-attention-work-e02fc6b942f64b2ba82ce7e35afc817d)
- [WEEK9: How does Multihead Self attention work?](https://www.notion.so/WEEK9-How-does-Multihead-Self-attention-work-cddce1ae09eb4b0fb067a2474cbf8515)
- [WEEK9: How does Residual Connection & Layer normalisation work?](https://www.notion.so/WEEK9-How-does-Residual-Connection-Layer-normalisation-work-b4a41db45a014378bc1c4a0f6da3757e)
- [WEEK9: How does Positional Encoding  work?](https://www.notion.so/WEEK9-How-does-Positional-Encoding-work-0d0e5b9d17464af08f39b4977c073beb)


## Quick Start

### Create a virtualenv and activate it 

```shell
conda create -n enkorde
conda activate enkorde
```

### Install Dependencies
```shell
pip3 install -r requirements.txt
```

### Define Configurations

`enkorde/config.yaml`을 입맛대로 변경하기:
```yaml
# --- config for training a model --- #
train:
  overfit:
    hidden_size: 512
    ffn_size: 512
    heads: 32
    depth: 3
    max_epochs: 10
    max_length: 145
    batch_size: 64
    lr: 0.0001
    tokenizer: wp
    dropout: 0.0
    seed: 410
    shuffle: true
    data: kor2eng
  # just to test things out
  overfit_small:
    hidden_size: 512
    ffn_size: 512
    heads: 32
    depth: 3
    max_epochs: 200
    max_length: 145
    batch_size: 64
    lr: 0.0001
    tokenizer: wp
    dropout: 0.0
    seed: 410
    # shuffle the training set
    shuffle: true
    data: kor2eng_small

# --- config for building a tokenizer --- #
build:
  vocab_size: 20000
  pad: "[PAD]"
  pad_id: 0
  unk: "[UNK]"
  unk_id: 1
  bos: "[BOS]"
  bos_id: 2
  eos: "[EOS]"
  eos_id: 3

```

### Build a tokenizer

학습에 사용할 토크나이저 구축하기 (WordPiece (`ver=wp`) 혹은 Byte Pair Encoding (`ver=bpe`) 중 선택):
```shell
# building a WordPiece tokenizer
python3 main_build.py --ver=wp
```

### Train a transformer

트랜스포머에게 한-영 번역을 End-to-End로 가르치기 (`config.yaml` 에서 정의한 버전을 선택):
```shell
# overfitting a transformer on a small dataset 
python3 main_train.py --ver=overfit_small
```

### Translate a Korean sentence with a pre-trained transformer

학습된 트랜스포머로 한글을 영어로 번역해보기:
```shell
python3 main_infer.py --ver=overfit_small --sent="..."
```
