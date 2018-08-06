

class Partition:

    def __init__(self):

        self.next_index = 0
        self.indices = []
        self.blocks = []

    def add(self, block):

        self.blocks.append(block)
        self.indices.append(self.next_index)
        self.next_index += 1

    def remove(self, index):

        self.indices.index(index)
        del self.blocks[index]
        del self.indices[index]

    def __eq__(self, other):

        return self.indices == other.indices

    def __ne__(self, other):

        return not self.__eq__(other)

    def __iter__(self):

        return zip(self.indices, self.blocks)

    def __len__(self):

        return len(self.indices)