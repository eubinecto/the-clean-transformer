from Korpora import Korpora
from dekorde.paths import KORPORA_DIR
from fetchers import fetch_kor2eng


def main():
    # fetching aihub does not work, sadly.
    Korpora.fetch('korean_parallel_koen_news',
                  root_dir=KORPORA_DIR)

    korpus = fetch_kor2eng()
    print(korpus.get_all_texts())

if __name__ == '__main__':
    main()
