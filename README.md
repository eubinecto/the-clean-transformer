# Enkorde

트랜스포머 밑바닥부터 구현 (광주인공지능사관학교/멋쟁이사자차럼 자연어처리 보충수업)

## Quick Start

### 프로젝트 클론 & 가상환경 구축
```shell
git clone https://github.com/eubinecto/enkorde.git
conda create -n enkorde python=3.9 
conda activate enkorde
conda install pip
cd enkorde
pip install -r requirements.txt
```

### 사전학습된 모델 다운로드 & 한국어 번역

`main_infer.py` 스크립트를 실행합니다. 사전학습된 모델을 다운로드 하기 위해선 반드시 첫번째 인자 (`entity`)로 `eubinecto`를 넣어야 합니다.
영어로 번역하고자 하는 한국어 문장을 `--kor` 인자로 넣어줍니다. 

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

3을 입력 후 엔터를 눌러주세요. 이후 사전학습된 트랜스포머 모델이 `enkorde/artifacts/transformer_scratch` 에 다운로드되며, 다음과 같이 주어진 :
```text
wandb: You chose 'Don't visualize my results'
wandb: Downloading large artifact transformer_torch:overfit_small, 262.36MB. 1 files... Done. 0:0:0
그러나 이것은 또한 책 ##상 ##도 필요로 하지 않는다 . -> like all op ##ti ##ca ##l mic ##e , but it also doesn ' t need a des ##k .
```


## 강의노트
- [WEEK7: Transformer - why?](https://www.notion.so/WEEK7-Transformer-why-8e3712fb674a4ba2a85bf6da9cd36af0)
- [WEEK8: Transformer: How does Self-attention work?](https://www.notion.so/WEEK8-Transformer-How-does-Self-attention-work-e02fc6b942f64b2ba82ce7e35afc817d)
- [WEEK9: How does Multihead Self attention work?](https://www.notion.so/WEEK9-How-does-Multihead-Self-attention-work-cddce1ae09eb4b0fb067a2474cbf8515)
- [WEEK9: How does Residual Connection & Layer normalisation work?](https://www.notion.so/WEEK9-How-does-Residual-Connection-Layer-normalisation-work-b4a41db45a014378bc1c4a0f6da3757e)
- [WEEK9: How does Positional Encoding  work?](https://www.notion.so/WEEK9-How-does-Positional-Encoding-work-0d0e5b9d17464af08f39b4977c073beb)

