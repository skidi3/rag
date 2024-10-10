
class Animal:
    def __init__(self, name, sound):
        self.name = name
        self.sound = sound

    def make_sound(self):
        return f"The {self.name} goes '{self.sound}'"

if __name__ == "__main__":
    cat = Animal("cat", "meow")
    print(cat.make_sound())
