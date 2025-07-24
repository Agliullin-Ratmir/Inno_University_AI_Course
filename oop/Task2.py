class NumberList:
    __amount = 0
    def __init__(self, number_list):
        self.number_list = number_list

    def count_local_maxima(self):
        for i in range(1, len(self.number_list) - 1):
            if self.number_list[i] > self.number_list[i - 1] and self.number_list[i] > self.number_list[i + 1]:
                self.__amount = self.__amount + 1
        return self.__amount


nl = NumberList([1, 5, 2, 4, 1])
print(nl.count_local_maxima())

