class NestedIterator:

    __result_list = []
    def __init__(self):
        pass

    def clear_result_list(self):
        self.__result_list = []

    def unwrap(self, incoming_list):
        if isinstance(incoming_list, list) == False:
            self.__result_list.append(incoming_list)
        else:
            for item in incoming_list:
                self.unwrap(item)
        return self.__result_list

nested = NestedIterator()
print(nested.unwrap(incoming_list=[[1,1],2,[1,1]]))
nested.clear_result_list()
print(nested.unwrap(incoming_list=[1,[4,[6]]]))