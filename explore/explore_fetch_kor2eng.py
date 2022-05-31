from cleanformer.fetchers import fetch_kor2eng


def main():
    fetch_kor2eng()
    train, val, test = fetch_kor2eng()


