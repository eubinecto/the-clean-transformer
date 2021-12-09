from Korpora import Korpora
from dekorde.paths import KORPORA_DIR


def main():
    # fetching aihub does not work, sadly.
    Korpora.fetch('aihub_spoken_translation',
                  root_dir=Korpora)


if __name__ == '__main__':
    main()
