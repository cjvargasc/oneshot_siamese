from training.train import Trainer
from testing.test import Tester


def train():
    Trainer.train()


def test():
    Tester.test()
    #Tester.test_one()


if __name__ == "__main__":
    train()
    test()


