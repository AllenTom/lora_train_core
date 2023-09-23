import argparse
import initapp

initapp.init_global()
from modules import doctor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train script checker')
    parser.add_argument('--json_out', action='store_true', help='Output as json')
    parser.add_argument('--case', type=str,default='all', help='case')
    args = parser.parse_args()
    doctor.check(
        json_out=args.json_out,

    )
