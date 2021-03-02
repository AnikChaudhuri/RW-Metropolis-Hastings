class Pet:
    def __init__(self,name,age):
        self.name=name
        self.age = age

    def show(self):
        print(f"I am {self.name} and I am {self.age} years old")

class Cat(Pet):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color

    def show(self):
        print(f"I am {self.name} and I am {self.age} years old and I am {self.color}")


    def speak(self):
        print("Meow")

class Dog(Pet):
    def speak(self):
        print("bark")

P = Pet("Tim",19)
P.show()
c= Cat("Bill", 20, "Brown")
c.show()
d = Dog("Jill",21)
d.show()
d.speak()
print(d.name)
