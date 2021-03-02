class person:
    number_of_person = 0
    def __init__(self, name):
        self.name = name
        person.number_of_person += 1

person.number_of_person = 8

class Math:
    
    @staticmethod
    def add(x):
        return x + 5

    @staticmethod
    def pr():
        print("run")

p1 = person("tim")
print(p1.number_of_person)
p2 = person("Bill")
print(p2.number_of_person)

print(Math.add(15))
Math.pr()
