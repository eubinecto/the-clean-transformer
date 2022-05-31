# The Clean Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/181hTrhfbmyub7UaMJmBY_fbFfLBCBi58?usp=sharing)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/eubinecto/the-clean-transformer.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/eubinecto/the-clean-transformer/context:python)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

🇰🇷 `pytorch-lightning`과 `wandb`로 깔끔하게 구현해보는 트랜스포머 

🇬🇧 Transformer implemented with clean and structured code - much thanks to `pytorch-lightning` & `wandb `!


## Quick Start
우선, 리포를 클론하고 가상환경을 구축합니다:
```shell
git clone https://github.com/eubinecto/the-clean-transformer.git
python3.9 -m venv venv
source venv/bin/activate
cd the-clean-transformer
pip3 install -r requirements.txt
```

이후 사전학습된 모델을 다운로드하고, 간단한 한국어 번역을 시도해보기 위해 `main_infer.py` 스크립트를 실행합니다. 
사전학습된 모델을 다운로드 하기 위해선 반드시 첫번째 인자 (`entity`)로 `eubinecto`를 넣어야 합니다.
추가로 영어로 번역하고자 하는 한국어 문장을 `--kor` 인자로 넣어줍니다. 
```shell
python3 main_infer.py eubinecto --kor="카페인은 원래 커피에 들어있는 물질이다."
```

위 스크립트를 실행하면, 다음과 같은 선택창이 뜹니다:
```text
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```

3을 입력 후 엔터를 눌러주세요. 이후 사전학습된 트랜스포머 모델이 `./artifacts/transformer:overfit_small` 에 다운로드되며, 다음과 같이 주어진
`--kor` 문장을 영어로 번역합니다:
```text
wandb: You chose 'Don't visualize my results'
wandb: Downloading large artifact transformer:overfit_small, 263.49MB. 1 files... Done. 0:0:0
카페 ##인은 원래 커피 ##에 들어 ##있는 물질 ##이다 . -> caf ##fe ##ine is a subst ##ance natural ##ly found in coffee .
```

## Pretrained Models

`overfit_small`
--- | 
데모를 위해 한국어-영어 말뭉치의 일부분만을 과학습한 모델 |
<img width="915" alt="image" src="https://user-images.githubusercontent.com/56193069/147040774-cabb3403-a07b-44f2-b759-6cd74dd16b6e.png"> |
[하이퍼파라미터](https://github.com/eubinecto/the-clean-transformer/blob/92d2e6e0e275af6cbb7b8d374bc2f7a3972615ac/config.yaml#L18-L32) / [학습말뭉치](https://github.com/eubinecto/the-clean-transformer/blob/92d2e6e0e275af6cbb7b8d374bc2f7a3972615ac/cleanformer/datamodules.py#L71-L82) / [Weights & Biases 학습로그](https://wandb.ai/eubinecto/cleanformer/runs/1ebht4yh/overview?workspace=user-eubinecto)|
use this in command: `python3 main_infer.py eubinecto --ver=overfit_small`|


## Project Structure 
```
.                        # ROOT_DIR 
├── main_build.py        # 주어진 말뭉치에 적합한 huggingface 토크나이저를 훈련시킬 때 사용하는 스크립트
├── main_train.py        # 구현된 트랜스포머를 훈련시킬 때 사용하는 스크립트
├── main_infer.py        # 사전학습된 트랜스포머로 예측을 해볼 때 사용하는 스크립트
├── config.yaml          # main_build.py 와 main_train.py에 필요한 인자를 정의해놓는 설정파일
└── cleanformer          # main 스크립트에 사용될 재료를 정의하는 파이썬 패키지
    ├── builders.py      # 말뭉치 -> 입력텐서, 정답텐서 변환을 도와주는 빌더 정의
    ├── functional.py    # 트랜스포머 구현에 필요한 텐서구축 함수 정의
    ├── datamodules.py   # 학습에 사용할 train/val/test 데이터 정의
    ├── models.py        # 모든 신경망 모델 정의
    ├── fetchers.py      # 데이터를 다운로드 및 로드하는 함수 정의
    ├── paths.py         # fetchers.py가 데이터를 다운로드 및 로드할 경로 정의
    └── __init__.py          
```
