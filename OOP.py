class Dog:

    def __init__(self,name, age):
        self.nam = name
        self.age = age
        print(self.nam)

    def add_one(self,x):
        return x+1

    def bark(self):
        print("bark")
        
    def get_name(self):
        return self.nam
    
    def get_age(self):
        return self.age

    def set_age(self,age):
        self.age = age

#d = Dog()
#d.bark()
#print(d.add_one(5))
#print(type(d))
e = Dog("Tim",34)
print(e.nam)
print(e.add_one(5))
e.bark()
e2 = Dog("Bill",12)
print(e2.nam)#Bill
print(e2.get_name())#Bill
print(e2.get_age())#12
e2.set_age(23)#set age to 23
print(e2.get_age())#23
