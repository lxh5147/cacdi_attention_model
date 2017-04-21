from keras.callbacks import Callback
try:
    import queue
except ImportError:
    import Queue as queue
from keras.engine.training import generator_queue
import time
import subprocess


class EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript(Callback):

    def __init__(self, val_generator, val_samples, filepath, save_best_only=False,  patience=0, verbose=0):
        super(EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript, self).__init__()
        self.filepath = filepath
        self.patience = patience
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.wait = 0
        self.val_generator = val_generator
        self.val_samples = val_samples
        self.optimal_threshold = 0.5


    def on_train_begin(self, logs={}):
        self.wait = 0       # Allow instances to be re-used
        self.best_acc = 0.0
        self.optimal_threshold = 0.0
        self.best_epoch = 0

    @staticmethod
    def _export(all_id, all_pred, all_true, pred_file, true_file):
        # enc_id,A41.9,D61.811,D62,D64.81,E43,E46,G92,G93.40,G93.41,I50.21,I50.22,I50.23,I50.31,I50.32,I50.33,I50.41,I50.42,I50.43,I50.9,J15.212,J15.6,J18.9,J44.1,J69.0,J96.00,J96.01,J96.02,J96.10,J96.11,J96.12,J96.20,J96.21,J96.22,K72.91,N17.9,R57.0,R57.1,R57.9,R65.20,R65.21
        # CHBG_H00056025406_20151012,0.00512823,0.000887801,0.00413668,0.000824932,0.00168439,0.000745712,0.00236619,0.000868717,0.00240495,0.000639154,0.00278882,0.00151651,0.00107024,0.00314594,0.00173389,0.000673811,0.00358504,0.000974634,0,0.00126461,0.000721962,0.00185213,0.00214863,0.0013375,0.000371074,0.000405095,0.000361469,0.000350338,0.000632457,0,0.000299509,0.00029951,0.000373421,0.000295824,0.0103653,0.000824228,0.00249404,0.000631504,0.00130254,0.00138589
        # all_pred and all_true: a list of numpy arrays
        output_dim = len(all_pred[0][0])
        code_list  = ['code_'+ str(i) for i in range(output_dim)]
        header = ','.join(code_list)
        header = ','.join(['enc_id',header])
        with open(pred_file,"w") as _f_pred:
            _f_pred.write(header + '\n')
            enc_id = 0
            for one_id, one_pred in zip(all_id,all_pred):
                for id, pred in zip(one_id, one_pred):
                    line = ','.join([str(v) for v in pred])
                    line = id + ',' + line
                    _f_pred.write(line + '\n')
                    enc_id += 1
        with open(true_file,"w") as _f_true:
            _f_true.write(header + '\n')
            enc_id = 0
            for one_id, one_true in zip(all_id,all_true):
                for id, _true in zip(one_id,one_true):
                    line = ','.join([str(int(v)) for v in _true])
                    line = id + ',' + line
                    _f_true.write(line + '\n')
                    enc_id += 1

    @staticmethod
    def _fine_tune(pred_file, true_file):
        # tune_threshold_daily.pl  dev.predict.csv ../output/partition.filtered/summit_ref.sequential.dev.csv '--grace 999 --suppress-repeated-fp --suppress-after-cds --suppress-weekend' <output scores per threshold> <dir for storing details of each threshold>
        script = 'tune_threshold_daily.pl'
        threshold_score_file = pred_file + '.threshold.scores'
        # assume the detail_dir will be created
        detail_dir = 'tmp'
        command =[script, pred_file, true_file, '--grace 999 --suppress-repeated-fp --suppress-after-cds --suppress-weekend',  threshold_score_file, detail_dir ]
        subprocess.check_call(command)

        # 0.085 0.109 0.079 0.092
        # threshold p r f1
        threshold_f1_list=[]
        with open(threshold_score_file,"r") as _f_score:
            for line in _f_score:
                line=line.rstrip('\n')
                parts = line.split(' ')
                threshold_f1_list.append((float(parts[0]),float(parts[-1])))

        sorted(threshold_f1_list, key=lambda threshold_f1: threshold_f1[1], reverse=True)
        return threshold_f1_list[0][0]


    @staticmethod
    def _evaluate(threshold, pred_file, true_file):
        script = 'daily_evaluation_script'
        threshold_option = '-t'
        ref_option = '--ref-csv'
        pred_option = '--hyp-csv'
        output_file = pred_file + '_' + str(threshold) + ".log"
        output_file_option = '--output'
        command = [script, '--grace', '999', '--suppress-repeated-fp', '--suppress-after-cds', '--suppress-weekend', threshold_option, str(threshold), ref_option,true_file, pred_option, pred_file, output_file_option,output_file]
        subprocess.check_call(command)
        '''
                            Ref    TP     FP     FN  Precision  Recall     F1
A41.9                66    10     97     56      0.093   0.152  0.116
D61.811               8     0      0      8      0.000   0.000  0.000
D62                  44     9     30     35      0.231   0.205  0.217
D64.81                6     0      0      5      0.000   0.000  0.000
E43                   4     0      5      4      0.000   0.000  0.000
E46                   3     0      0      3      0.000   0.000  0.000
G92                  17     0      3     17      0.000   0.000  0.000
G93.40                0     0      0      0      0.000   0.000  0.000
G93.41               52     6     19     46      0.240   0.115  0.156
I50.21                2     0      0      2      0.000   0.000  0.000
I50.22               14     1     11     13      0.083   0.071  0.077
I50.23               10     0      0     10      0.000   0.000  0.000
I50.31                4     0      0      4      0.000   0.000  0.000
I50.32               30     2      2     28      0.500   0.067  0.118
I50.33               21     0      1     21      0.000   0.000  0.000
I50.41                3     0      0      3      0.000   0.000  0.000
I50.42               15     0      0     15      0.000   0.000  0.000
I50.43               10     0      0     10      0.000   0.000  0.000
I50.9                 1     0      0      1      0.000   0.000  0.000
J15.212               6     0      0      6      0.000   0.000  0.000
J15.6                 3     0      0      3      0.000   0.000  0.000
J18.9                 9     0      0      9      0.000   0.000  0.000
J44.1                 6     0      0      6      0.000   0.000  0.000
J69.0                 5     0      0      5      0.000   0.000  0.000
J96.00               15     0      0     15      0.000   0.000  0.000
J96.01               26     0      0     26      0.000   0.000  0.000
J96.02                4     0      0      4      0.000   0.000  0.000
J96.10                4     0      0      4      0.000   0.000  0.000
J96.11                8     0      0      8      0.000   0.000  0.000
J96.12                2     0      0      2      0.000   0.000  0.000
J96.20                0     0      0      0      0.000   0.000  0.000
J96.21                3     0      0      3      0.000   0.000  0.000
J96.22                3     0      0      3      0.000   0.000  0.000
K72.91                0     0      0      0      0.000   0.000  0.000
N17.9                27     3     41     24      0.068   0.111  0.085
R57.0                 1     0      0      1      0.000   0.000  0.000
R57.1                 7     0      0      7      0.000   0.000  0.000
R57.9                 0     0      0      0      0.000   0.000  0.000
R65.20               57     0      2     57      0.000   0.000  0.000
R65.21               10     0      0     10      0.000   0.000  0.000
Total (micro-avg)   506    31    211    474      0.128   0.061  0.083
Total (macro-avg)   506    31    211    474      0.030   0.018  0.019
        '''
        with open(output_file,'r') as _f_log:
            lines = _f_log.readlines()
            # get the micro-avg
            line = lines[-2]
            return float(line.split()[-1])

    @staticmethod
    def evaluate_with_external_scripts(model, data_generator_with_id, total_nb_samples, filepath, optimal_threshold=None):
        processed_samples = 0
        wait_time = 0.01
        all_pred = []
        all_true = []
        all_id = []

        data_gen_queue, _stop, generator_threads = generator_queue(data_generator_with_id)

        while processed_samples < total_nb_samples:
            generator_output = None
            while not _stop.is_set():
                if not data_gen_queue.empty():
                    generator_output = data_gen_queue.get()
                    break
                else:
                    time.sleep(wait_time)

            if not hasattr(generator_output, '__len__'):
                _stop.set()
                raise Exception('output of generator should be a tuple '
                                '(x, y, sample_weight) '
                                'or (x, y). Found: ' + str(generator_output))
            if len(generator_output) == 3:
                x, y, id = generator_output
                sample_weight = None
            elif len(generator_output) == 4:
                x, y, id, sample_weight = generator_output
            else:
                _stop.set()
                raise Exception('output of generator should be a tuple '
                                '(x, y, sample_weight) '
                                'or (x, y). Found: ' + str(generator_output))

            all_id.append(id)
            try:
                outs = model.predict_on_batch(x)
            except:
                _stop.set()
                raise

            if type(x) is list:
                nb_samples = len(x[0])
            elif type(x) is dict:
                nb_samples = len(list(x.values())[0])
            else:
                nb_samples = len(x)

            # y and out: mini_batch_size, 1, output_dim
            all_true.append(y[:,0,:])
            all_pred.append(outs[:,0,:])

            processed_samples += nb_samples

        pred_file = filepath + ".pred.csv"
        true_file = filepath + ".true.csv"
        EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript._export(all_id, all_pred, all_true,pred_file, true_file)
        if optimal_threshold is None:
            optimal_threshold = EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript._fine_tune(pred_file, true_file)
        return optimal_threshold, EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript._evaluate(optimal_threshold, pred_file,
                                                                                        true_file)
    def evaluate(self, epoch, logs):
        filepath = self.filepath.format(epoch=epoch, **logs)
        return EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript.evaluate_with_external_scripts(self.model,self.val_generator, self.val_samples, filepath)

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)

        optimal_threshold, current_acc = self.evaluate(epoch,logs)
        if current_acc > self.best_acc:
            if self.verbose > 0:
                print('Epoch %05d: improved from %0.5f to %0.5f,'
                      ' saving model to %s'
                      % (epoch, self.best_acc,
                         current_acc, filepath))
            self.best_acc = current_acc
            self.best_epoch = epoch
            # track the optimal threshold
            self.optimal_threshold = optimal_threshold
            self.wait = 0
            self.model.save_weights(filepath, overwrite=True)
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch %05d: early stopping' % (epoch))
                self.model.stop_training = True
            self.wait += 1
            if not self.save_best_only:
                self.model.save_weights(filepath, overwrite=True)