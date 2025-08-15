class Singleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if cls.instance == None:
            cls.instance = super().__call__(*args, **kwargs)
        return cls.instance

class SomeClass(metaclass=Singleton):
    def __init__(self, value=None):
        self.value = value

    def __repr__(self):
        return f"Some class with value: {self.value}"

    def get_value(self):
        return self.value

if __name__ == "__main__":
    obj1 = SomeClass(10)
    obj2 = SomeClass(20)
    obj3 = SomeClass(30)

    assert obj2 is obj1
    assert obj2.get_value() == obj1.get_value()

    assert obj3 is obj1
    assert obj3.get_value() == obj1.get_value()

    print(obj1)
    print(obj2)
    print(obj3)