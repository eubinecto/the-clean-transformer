# Enkorde
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/181hTrhfbmyub7UaMJmBY_fbFfLBCBi58?usp=sharing)

[Weights& Biases](https://wandb.ai/eubinecto/enkorde/artifacts/dataset/seoul2jeju/07c261d01dff37c66a8c)와
[pytorch-lightning](https://www.pytorchlightning.ai)으로 밑바닥부터 구현해보는 트랜스포머 
> [광주인공지능사관학교](https://aischool.likelion.net) X [멋쟁이사자차럼](https://www.likelion.net) 2기 자연어처리 과정 수업자료
## Quick Start
우선, 리포를 클론하고 가상환경을 구축합니다:
```shell
git clone https://github.com/eubinecto/enkorde.git
conda create -n enkorde python=3.9 
conda activate enkorde
conda install pip
cd enkorde
pip install -r requirements.txt
```

이후 사전학습된 모델을 다운로드하고, 간단한 한국어 번역을 시도해보기 위해 `main_infer.py` 스크립트를 실행합니다. 
사전학습된 모델을 다운로드 하기 위해선 반드시 첫번째 인자 (`entity`)로 `eubinecto`를 넣어야 합니다.
추가로 영어로 번역하고자 하는 한국어 문장을 `--kor` 인자로 넣어줍니다. 
```shell
python3 main_infer.py eubinecto --kor="그러나 이것은 또한 책상도 필요로 하지 않는다."
```

위 스크립트를 실행하면, 다음과 같은 선택창이 뜹니다:
```text
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```

3을 입력 후 엔터를 눌러주세요. 이후 사전학습된 트랜스포머 모델이 `enkorde/artifacts/transformer:overfit_small` 에 다운로드되며, 다음과 같이 주어진
`--kor` 문장을 영어로 번역합니다:
```text
wandb: You chose 'Don't visualize my results'
wandb: Downloading large artifact transformer_torch:overfit_small, 262.36MB. 1 files... Done. 0:0:0
그러나 이것은 또한 책 ##상 ##도 필요로 하지 않는다 . -> like all op ##ti ##ca ##l mic ##e , but it also doesn ' t need a des ##k .
```

## 프로젝트 구조 
```
.                        # ROOT_DIR 
├── main_build.py        # 주어진 말뭉치에 적합한 huggingface 토크나이저를 훈련시킬 때 사용하는 스크립트
├── main_train.py        # 구현된 트랜스포머를 훈련시킬 때 사용하는 스크립트
├── main_infer.py        # 사전학습된 트랜스포머로 예측을 해볼 때 사용하는 스크립트
├── config.yaml          # main_build.py 와 main_train.py에 필요한 인자를 정의해놓는 설정파일
└── enkorde              # main 스크립트에 사용될 재료를 정의하는 파이썬 패키지
    ├── builders.py      # 말뭉치 -> 입력텐서, 정답텐서 변환을 도와주는 빌더 정의
    ├── tensors.py       # 트랜스포머 구현에 필요한 상수텐서 정의 (e.g. subsequent_mask, positional_encoding)
    ├── datamodules.py   # 학습에 사용할 train/val/test 데이터 정의
    ├── models.py        # 모든 신경망 모델 정의
    ├── fetchers.py      # 데이터를 다운로드 및 로드하는 함수 정의
    ├── paths.py         # fetchers.py가 데이터를 다운로드 및 로드할 경로 정의
    └── __init__.py          
```

## 트랜스포머란?
- [WEEK7: Transformer - why?](https://www.notion.so/WEEK7-Transformer-why-8e3712fb674a4ba2a85bf6da9cd36af0)
- [WEEK8: Transformer: How does Self-attention work?](https://www.notion.so/WEEK8-Transformer-How-does-Self-attention-work-e02fc6b942f64b2ba82ce7e35afc817d)
- [WEEK9: How does Multihead Self attention work?](https://www.notion.so/WEEK9-How-does-Multihead-Self-attention-work-cddce1ae09eb4b0fb067a2474cbf8515)
- [WEEK9: How does Residual Connection & Layer normalisation work?](https://www.notion.so/WEEK9-How-does-Residual-Connection-Layer-normalisation-work-b4a41db45a014378bc1c4a0f6da3757e)
- [WEEK9: How does Positional Encoding  work?](https://www.notion.so/WEEK9-How-does-Positional-Encoding-work-0d0e5b9d17464af08f39b4977c073beb)

