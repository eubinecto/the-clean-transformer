from cleanformer.fetchers import fetch_kor2eng


def main():
    train, val, test = fetch_kor2eng("kor2eng:v0")
    print(len(train))
    print(len(val))
    print(len(test))


if __name__ == '__main__':
    main()
