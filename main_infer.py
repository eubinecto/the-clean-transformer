"""
Just a simple script for demonstrating translation
"""
import argparse
from cleanformer import preprocess as P  # noqa
from cleanformer.translator import Translator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kor", type=str, default="집이 정말 편안하네요")
    args = parser.parse_args()
    translator = Translator()
    kor, eng = translator(args.kor)
    print(kor, "->", eng)


if __name__ == "__main__":
    main()
