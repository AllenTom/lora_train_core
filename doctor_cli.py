import argparse

from modules import doctor

# Read the requirements.txt file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train script checker')

    doctor.check()
