class Animal:
    def __init__(self, _name):
        self._name = _name
    def speak(self):
        return f"{self._name} издает звук"

class Dog(Animal):
    __mood = ""
    def get_mood(self):
        return self.__mood

    def set_mood(self, mood):
        if mood == "happy" or mood == "sad":
            self.__mood = mood
    def __init__(self, _name):
        super().__init__(_name)
        self._name = _name
    def speak(self):
        return "Гав"


class Cat(Animal):
    __mood = ""
    def get_mood(self):
        return self.__mood

    def set_mood(self, mood):
        if mood == "happy" or mood == "sad":
            self.__mood = mood

    def __init__(self, _name):
        self._name = _name
    def speak(self):
        return "Мяу"


dog = Dog("Шарик")
cat = Cat("Том")

pets = [dog, cat]

for animal in pets:
    print(animal.speak())