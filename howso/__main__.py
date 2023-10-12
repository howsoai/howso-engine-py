import argparse
from argparse import RawTextHelpFormatter
from datetime import date
import importlib.metadata
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '--version', dest='version', action='store_const', const=True,
        required=False, default=False,
        help='Display the version and quit.'
    )

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    else:
        args = parser.parse_args()
        if args.version:
            print(f"""
Howso (tm) client version: {importlib.metadata.version('howso-engine')}
Copyright (c) 2018-{date.today().year}, Howso Incorporated. All rights reserved.
""")
