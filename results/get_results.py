#!/usr/bin/env python

import argparse
import io
import logging
import os
import re
from collections import defaultdict

import sys

logger = logging.getLogger(__name__)


RE_EPOCH_LINE = re.compile('^Epoch ')
RE_EPOCH_LINE_END = re.compile('/.*$')
RE_VAL_ACC_LINE_START = re.compile('^.*val_acc: ')
RE_VAL_ACC_LINE_END = re.compile('INFO:root.*$')
RE_EVAL_RESULT_LINE = re.compile('^evaluation results:')
RE_EVAL_RESULT_LINE_START = re.compile('^.*, ')

RE_BEFORE_LR = re.compile('^.*lr_')
RE_AFTER_LR = re.compile('_.*$')
RE_BEFORE_DROPOUT = re.compile('^.*dropout_')
RE_AFTER_DROPOUT = re.compile('.log$')
RE_BEFORE_SEED = re.compile('^.*seed_')
RE_AFTER_SEED = re.compile('_.*$')


RE_N3LU_VAL_ACC = re.compile('INFO:n3lu.training:Detailed results on validation set:')
RE_N3LU_TEST_ACC = re.compile('INFO:n3lu.training:Detailed results on test set:')
RE_N3LU_VAL_ACC_NUMBER = re.compile('\d+\.\d+%')
RE_N3LU_LEARNING_RATE = re.compile('learning.rate:')
RE_N3LU_LEARNING_RATE_NUMBERS = re.compile('base:.*proj: \d+\.\d+')

RE_N3LU_SEED = re.compile('model.random_seed:')
RE_N3LU_SEED_NUMBERS = re.compile('\d+')


class Result:

    def __init__(self, val_accs, test_accs, lr, seeds, logs, dropout='none'):
        self.val_accs = val_accs
        self.test_accs = test_accs
        self.lr = lr
        self.seeds = seeds
        self.logs = logs
        self.dropout = dropout

    @staticmethod
    def format(f):
        return '{:03.1f}'.format(f)

    @staticmethod
    def format_pair(p):
        return '{} / {}'.format(Result.format(p[0]), Result.format(p[1]))

    @staticmethod
    def avg(accs):
        return sum(accs) / float(len(accs))

    def print_as_single(self):
        assert len(self.val_accs) == 1
        assert len(self.test_accs) == 1
        assert len(self.logs) == 1
        assert len(self.seeds) == 1
        return 'val acc / test_acc: {} - lr: {} - dropout: {} - seed: {} - log {}'.format(
                    Result.format_pair((self.val_accs[0], self.test_accs[0])),
                    self.lr, self.dropout, self.seeds[0], self.logs[0])

    def print_average(self, logs=True, add_all_scores=False):
        result = 'val acc / test_acc: {} - lr: {} - dropout: {} -  logs: {}'.format(
                    Result.format_pair((Result.avg(self.val_accs), Result.avg(self.test_accs))),
                    self.lr, self.dropout, ' '.join(self.logs) if logs else len(self.logs))
        if add_all_scores:
            result += '\n\t(' + ' -- '.join([Result.format_pair(x) for x in zip(self.val_accs, self.test_accs)]) + ')'
        return result

    @staticmethod
    def average_on_seeds(results):
        results_with_same_hp = defaultdict(lambda: [])
        for result in results:
            results_with_same_hp[(result.lr, result.dropout)].append(result)

        results = []
        for t, t_ress in results_with_same_hp.iteritems():
            val_all = []
            test_all = []
            logs = []
            seeds_all = []
            for t_res in t_ress:
                val_all.extend(t_res.val_accs)
                test_all.extend(t_res.test_accs)
                logs.extend(t_res.logs)
                seeds_all.extend(t_res.seeds)
            results.append(Result(val_all, test_all, t[0], seeds_all, logs, dropout=t[1]))
        return results

def parse_n3lu_logs(logs):
    results = []

    count = 0
    skipped = 0
    for log in logs:
        count += 1
        with io.open(log, 'r') as in_stream:
            if not in_stream.readlines()[-1].startswith('INFO:n3lu.training:Best validation error'):
                print('skipping {}'.format(log))
                skipped += 1
                continue
        with io.open(log, 'r') as in_stream:
            print('doing {} of {} -- {}'.format(count, len(logs), log))
            best_eval = -1
            best_test_eval = -1
            lr = -1
            seed = -1
            for line in in_stream:
                if RE_N3LU_VAL_ACC.match(line):
                    line = in_stream.readline()
                    best_eval = float(RE_N3LU_VAL_ACC_NUMBER.findall(line)[0].replace('%', ''))
                elif RE_N3LU_TEST_ACC.match(line):
                    line = in_stream.readline()
                    best_test_eval = float(RE_N3LU_VAL_ACC_NUMBER.findall(line)[0].replace('%', ''))
                elif RE_N3LU_LEARNING_RATE.match(line):
                    lr = RE_N3LU_LEARNING_RATE_NUMBERS.findall(line)[0].replace('%', '').replace(' ', '')
                elif RE_N3LU_SEED.match(line):
                    seed = RE_N3LU_SEED_NUMBERS.findall(line)[0].replace('%', '').replace(' ', '')
            to_append = Result([best_eval], [best_test_eval], lr, [seed], [log])
            results.append(to_append)
    print('done {} logs - skipped {}'.format(count - skipped, skipped))
    return results


def parse_keras_log(log):
    with io.open(log, 'r') as in_stream:
        val_acc_and_epoch = []
        lr = RE_AFTER_LR.sub('', RE_BEFORE_LR.sub('', log))
        seed = RE_AFTER_SEED.sub('', RE_BEFORE_SEED.sub('', log))
        if RE_BEFORE_DROPOUT.match(log):
            dropout = RE_AFTER_DROPOUT.sub('', RE_BEFORE_DROPOUT.sub('', log))
            if dropout == '': dropout = 'none'
        else:
            dropout = 'none'
        for line in in_stream:
            if RE_EPOCH_LINE.match(line):
                epoch = int(RE_EPOCH_LINE_END.sub('', RE_EPOCH_LINE.sub('', line)).strip())
                line_skipped = 0
                while not RE_VAL_ACC_LINE_START.match(line):
                    line = in_stream.readline()
                    line_skipped += 1
                    if line_skipped > 100000:
                        raise ValueError('too many lines skipped!')
                val_acc = float(RE_VAL_ACC_LINE_END.sub(
                    '', RE_VAL_ACC_LINE_START.sub('', line).strip()))
                val_acc_and_epoch.append((val_acc, epoch))
            elif RE_EVAL_RESULT_LINE.match(line):
                line = in_stream.readline()
                best_test_eval = float(RE_EVAL_RESULT_LINE_START.sub('', line).replace(']', '').strip())
        best_eval, best_epoch = sorted(val_acc_and_epoch)[-1]
        to_append = Result([best_eval * 100], [best_test_eval * 100], lr, [seed], [log], dropout=dropout)
        return to_append


def parse_keras_logs(logs):
    results = []
    count = 0
    skipped = 0
    for log in logs:
        count += 1
        try:
            results.append(parse_keras_log(log))
            print('parsed {} of {} -- {}'.format(count, len(logs), log))
        except ValueError:
            skipped += 1
            print('skipping {}'.format(log))
    print('done {} logs - skipped {}'.format(count - skipped, skipped))
    return results


def remove_seed(r):
    return (r[1], r[3])

def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--debug', help='keep/print debug info',
                        action='store_true')
    parser.add_argument("logs", help="log files",
                        nargs='+')
    parser.add_argument("--output", help="output",
                        required=True)
    parser.add_argument("--n3lu", help="parses n3lu files",
                        action='store_true')
    parser.add_argument("--no-sort", help="no sort performed on results (log file will be sorted instead)",
                        action='store_true')
    parser.add_argument("--no-seed-average", help="will keep all the seed "
                        " results, instead of printing the average",
                        action='store_true')
    parser.add_argument("--add-logs", help="will not print the logs",
                        action='store_true')
    parser.add_argument("--details", help="will print all scores",
                        action='store_true')

    args = parser.parse_args(argv[1:])

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if len(args.logs) < 1:
        print('no log file found!!')
        sys.exit(1)
    print('got {} log files'.format(len(args.logs)))

    if args.n3lu:
        print('parsing n3lu files')
        results = parse_n3lu_logs(args.logs)
    else:
        print('parsing keras files')
        results = parse_keras_logs(args.logs)

    if not args.no_seed_average:
        print('computing seed average for {} results'.format(len(results)))
        results = Result.average_on_seeds(results)
        
    if args.no_sort:
        print('sorting on lr / dropout')
        results = sorted(results, key=lambda x: (x.lr, x.dropout))
    else:
        print('sorting on validation results')
        results = sorted(results, key=lambda x: Result.avg(x.val_accs), reverse=True)

    with io.open(args.output, 'wb') as out_stream:
        for result in results:
            if args.no_seed_average:
                out_stream.write(result.print_as_single() + '\n')
            else:
                out_stream.write(result.print_average(
                        logs=args.add_logs, add_all_scores=args.details) + '\n')

if __name__ == '__main__':
    main()
