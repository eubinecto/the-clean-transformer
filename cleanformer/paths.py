"""
os.path.join 보단, 그냥 pathlib를 활용해서, 객체 지행적으로 코드를 짜는 편이 더 나을 것 같다.
Path 객체로 경로를 다루면, 나눗셈 오퍼레이터, "/", 를 os.join의 기능으로 사용할 수 있다.
코드가 더 보기도 좋고, 코드를 짜는 것도 한결 더 간편해짐.
"""

from pathlib import Path

# directories
ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
WANDB_DIR = ROOT_DIR / "wandb"
KORPORA_DIR = ROOT_DIR / "korpora"

# --- local files --- #
CONFIG_YAML = ROOT_DIR / "config.yaml"


# --- artifacts --- #
def tokenizer_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"tokenizer-{ver}"


def transformer_dir(ver: str) -> Path:
    return ARTIFACTS_DIR / f"transformer-{ver}"
