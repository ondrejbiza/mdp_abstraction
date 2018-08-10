import pickle


def read_pickle(path):
    """
    Read pickle from a file.
    :param path:    Path to pickle.
    :return:        Content of the pickle.
    """

    with open(path, "rb") as file:
        return pickle.load(file)


def write_pickle(path, data):
    """
    Write pickle to a file.
    :param path:    Path where to write the pickle.
    :param data:    Data to pickle.
    :return:        None.
    """

    with open(path, "wb") as file:
        pickle.dump(data, file)
