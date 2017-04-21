#!/usr/bin/env python

import argparse
import io
import logging
import os
import re

import sys

logger = logging.getLogger(__name__)


RE_EPOCH_LINE = re.compile('^Epoch ')
RE_VAL_ACC_LINE_START = re.compile('^.*val_acc: ')
RE_TO_FIRST_ACC = re.compile('^.*- acc: ')
RE_FLOAT = re.compile('^\d+\.\d+')
RE_TO_VAL_ACC = re.compile('^.*val_acc: ')

def parse_n3lu_logs(logs):
    pass


def print_diff(curr, old):
    if old is None:
        return '-----'
    diff = curr - old
    perc_diff = (diff / old) * 100
    return '{:+05.1f}'.format(perc_diff)


def parse_keras_logs(log):
    with io.open(log, 'r') as in_stream:
           old_train = old_val = None
           for line in in_stream:
               if RE_EPOCH_LINE.match(line):
                    line_skipped = 0
                    while not RE_VAL_ACC_LINE_START.match(line):
                        line = in_stream.readline()
                        line_skipped += 1
                        if line_skipped > 100000:
                            print('log file truncated!')
                            sys.stdout.flush()
                            raise ValueError('too many lines skipped!')
                    to_first_acc = RE_TO_FIRST_ACC.sub('', line)
                    train_acc = float(RE_FLOAT.findall(to_first_acc)[0])
                    to_val_acc = RE_TO_VAL_ACC.sub('', to_first_acc)
                    val_acc = float(RE_FLOAT.findall(to_val_acc)[0])
                    train_diff = print_diff(train_acc, old_train)
                    val_diff = print_diff(val_acc, old_val)
                    old_train = train_acc
                    old_val = val_acc
                    print('train: {:04.3f}  ({}) ---- val: {:04.3f}  ({})'.format(
                        train_acc, train_diff , val_acc, val_diff))


def remove_seed(r):
    return (r[1], r[3])

def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--debug', help='keep/print debug info',
                        action='store_true')
    parser.add_argument("logs", help="log file", nargs="+")
    parser.add_argument("--n3lu", help="parses n3lu files",
                        action='store_true')

    args = parser.parse_args(argv[1:])

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    print('parsing {} log files'.format(len(args.logs)))

    if args.n3lu:
        print('parsing n3lu files')
        #parse_n3lu_logs(args.logs)
    else:
        for log in args.logs:
            print('parsing keras log {}'.format(log))
            try:
                parse_keras_logs(log)
            except:
                pass

if __name__ == '__main__':
    main()
