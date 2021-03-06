# The Clean Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/181hTrhfbmyub7UaMJmBY_fbFfLBCBi58?usp=sharing)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/eubinecto/the-clean-transformer.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/eubinecto/the-clean-transformer/context:python)

๐ฐ๐ท `pytorch-lightning`๊ณผ `wandb`๋ก ๊น๋ํ๊ฒ ๊ตฌํํด๋ณด๋ ํธ๋์คํฌ๋จธ 

๐ฌ๐ง Transformer implemented with clean and structured code - much thanks to `pytorch-lightning` & `wandb `!


## Quick Start
์ฐ์ , ๋ฆฌํฌ๋ฅผ ํด๋ก ํ๊ณ  ๊ฐ์ํ๊ฒฝ์ ๊ตฌ์ถํฉ๋๋ค:
```shell
git clone https://github.com/eubinecto/the-clean-transformer.git
python3.9 -m venv venv
source venv/bin/activate
cd the-clean-transformer
pip3 install -r requirements.txt
```

์ดํ ์ฌ์ ํ์ต๋ ๋ชจ๋ธ์ ๋ค์ด๋ก๋ํ๊ณ , ๊ฐ๋จํ ํ๊ตญ์ด ๋ฒ์ญ์ ์๋ํด๋ณด๊ธฐ ์ํด `main_infer.py` ์คํฌ๋ฆฝํธ๋ฅผ ์คํํฉ๋๋ค. 
์ฌ์ ํ์ต๋ ๋ชจ๋ธ์ ๋ค์ด๋ก๋ ํ๊ธฐ ์ํด์  ๋ฐ๋์ ์ฒซ๋ฒ์งธ ์ธ์ (`entity`)๋ก `eubinecto`๋ฅผ ๋ฃ์ด์ผ ํฉ๋๋ค.
์ถ๊ฐ๋ก ์์ด๋ก ๋ฒ์ญํ๊ณ ์ ํ๋ ํ๊ตญ์ด ๋ฌธ์ฅ์ `--kor` ์ธ์๋ก ๋ฃ์ด์ค๋๋ค. 
```shell
python3 main_infer.py eubinecto --kor="์นดํ์ธ์ ์๋ ์ปคํผ์ ๋ค์ด์๋ ๋ฌผ์ง์ด๋ค."
```

์ ์คํฌ๋ฆฝํธ๋ฅผ ์คํํ๋ฉด, ๋ค์๊ณผ ๊ฐ์ ์ ํ์ฐฝ์ด ๋น๋๋ค:
```text
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```

3์ ์๋ ฅ ํ ์ํฐ๋ฅผ ๋๋ฌ์ฃผ์ธ์. ์ดํ ์ฌ์ ํ์ต๋ ํธ๋์คํฌ๋จธ ๋ชจ๋ธ์ด `./artifacts/transformer:overfit_small` ์ ๋ค์ด๋ก๋๋๋ฉฐ, ๋ค์๊ณผ ๊ฐ์ด ์ฃผ์ด์ง
`--kor` ๋ฌธ์ฅ์ ์์ด๋ก ๋ฒ์ญํฉ๋๋ค:
```text
wandb: You chose 'Don't visualize my results'
wandb: Downloading large artifact transformer:overfit_small, 263.49MB. 1 files... Done. 0:0:0
์นดํ ##์ธ์ ์๋ ์ปคํผ ##์ ๋ค์ด ##์๋ ๋ฌผ์ง ##์ด๋ค . -> caf ##fe ##ine is a subst ##ance natural ##ly found in coffee .
```

## Pretrained Models

`overfit_small`
--- | 
๋ฐ๋ชจ๋ฅผ ์ํด ํ๊ตญ์ด-์์ด ๋ง๋ญ์น์ ์ผ๋ถ๋ถ๋ง์ ๊ณผํ์ตํ ๋ชจ๋ธ |
<img width="915" alt="image" src="https://user-images.githubusercontent.com/56193069/147040774-cabb3403-a07b-44f2-b759-6cd74dd16b6e.png"> |
[ํ์ดํผํ๋ผ๋ฏธํฐ](https://github.com/eubinecto/the-clean-transformer/blob/92d2e6e0e275af6cbb7b8d374bc2f7a3972615ac/config.yaml#L18-L32) / [ํ์ต๋ง๋ญ์น](https://github.com/eubinecto/the-clean-transformer/blob/92d2e6e0e275af6cbb7b8d374bc2f7a3972615ac/cleanformer/datamodules.py#L71-L82) / [Weights & Biases ํ์ต๋ก๊ทธ](https://wandb.ai/eubinecto/cleanformer/runs/1ebht4yh/overview?workspace=user-eubinecto)|
use this in command: `python3 main_infer.py eubinecto --ver=overfit_small`|


## Project Structure 
```
.                        # ROOT_DIR 
โโโ main_build.py        # ์ฃผ์ด์ง ๋ง๋ญ์น์ ์ ํฉํ huggingface ํ ํฌ๋์ด์ ๋ฅผ ํ๋ จ์ํฌ ๋ ์ฌ์ฉํ๋ ์คํฌ๋ฆฝํธ
โโโ main_train.py        # ๊ตฌํ๋ ํธ๋์คํฌ๋จธ๋ฅผ ํ๋ จ์ํฌ ๋ ์ฌ์ฉํ๋ ์คํฌ๋ฆฝํธ
โโโ main_infer.py        # ์ฌ์ ํ์ต๋ ํธ๋์คํฌ๋จธ๋ก ์์ธก์ ํด๋ณผ ๋ ์ฌ์ฉํ๋ ์คํฌ๋ฆฝํธ
โโโ config.yaml          # main_build.py ์ main_train.py์ ํ์ํ ์ธ์๋ฅผ ์ ์ํด๋๋ ์ค์ ํ์ผ
โโโ cleanformer          # main ์คํฌ๋ฆฝํธ์ ์ฌ์ฉ๋  ์ฌ๋ฃ๋ฅผ ์ ์ํ๋ ํ์ด์ฌ ํจํค์ง
    โโโ builders.py      # ๋ง๋ญ์น -> ์๋ ฅํ์, ์ ๋ตํ์ ๋ณํ์ ๋์์ฃผ๋ ๋น๋ ์ ์
    โโโ functional.py    # ํธ๋์คํฌ๋จธ ๊ตฌํ์ ํ์ํ ํ์๊ตฌ์ถ ํจ์ ์ ์
    โโโ datamodules.py   # ํ์ต์ ์ฌ์ฉํ  train/val/test ๋ฐ์ดํฐ ์ ์
    โโโ models.py        # ๋ชจ๋  ์ ๊ฒฝ๋ง ๋ชจ๋ธ ์ ์
    โโโ fetchers.py      # ๋ฐ์ดํฐ๋ฅผ ๋ค์ด๋ก๋ ๋ฐ ๋ก๋ํ๋ ํจ์ ์ ์
    โโโ paths.py         # fetchers.py๊ฐ ๋ฐ์ดํฐ๋ฅผ ๋ค์ด๋ก๋ ๋ฐ ๋ก๋ํ  ๊ฒฝ๋ก ์ ์
    โโโ __init__.py          
```
