
import random

def roll_dice():
    return random.randint(1, 6)

if __name__ == "__main__":
    print(f"Rolled a dice: {roll_dice()}")
