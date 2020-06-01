import numpy as np 

class Test(object):
    def __init__(self):
        self.a = "Init"
        self.b = 25

    # @classmethod
    # def add11(cls, input):
    #     return input + self.b

    def __call__(self, input):
        print(self.a)
        return input + self.b

test = Test()
b = test(3)
print(b)


    

    