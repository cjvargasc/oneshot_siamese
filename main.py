from training.train import Trainer
#from training.train_test import train_model
from testing.test import Tester
#from testing.TestFullImages import Tester


def train():
    Trainer.train()
    #train_model()

def test():
    Tester.test()

if __name__ == "__main__":
    #train()
    test()


