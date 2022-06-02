import numpy


def main():
    x_1 = numpy.zeros((10, 7))
    x_2 = numpy.zeros((10, 7))
    x = numpy.concatenate([x_1, x_2], axis=0)
    print(x.shape)


if __name__ == "__main__":
    main()
