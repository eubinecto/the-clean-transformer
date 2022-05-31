from Korpora import Korpora
from cleanformer.paths import KORPORA_DIR


def main():
    # fetching aihub does not work, sadly.
    Korpora.fetch('korean_parallel_koen_news',
                  root_dir=KORPORA_DIR)


if __name__ == '__main__':
    main()
