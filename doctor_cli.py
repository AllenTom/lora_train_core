import argparse

from modules import doctor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train script checker')

    doctor.check()
