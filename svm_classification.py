from __future__ import division
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

from bgpanomalies import *
from lstm_detection import *
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

def compute_weights(xy_total):
    y_total = xy_total[:,-1:]
    y_weights = to_categorical(y_total, num_classes=4)
    y_weights = y_weights.reshape(-1, 4)
    y_classes = pd.DataFrame(y_weights).idxmax(1, skipna=False)
    label_encoder = LabelEncoder()
    label_encoder.fit(list(y_classes))
    y_integers = label_encoder.transform(list(y_classes))
    sample_weights = compute_sample_weight('balanced', y_integers)
    return sample_weights

def print_metrics(y_pred, accuracy, test_file):
    count = Counter(y_pred)
    length = len(y_pred)
    indirect = np.round(count[1]/len(y_pred)*100,2)
    direct = np.round(count[2]/len(y_pred)*100,2)
    outage = np.round(count[3]/len(y_pred)*100,2)
    accuracy = np.round(accuracy*100,2)
    # print test_file
    # print 'len      -> '  + str(length)
    # print 'indirect -> '  + str(indirect) + '%'
    # print 'direct   -> '  + str(direct) + '%'
    # print 'outage   -> '  + str(outage) + '%'
    print '*** accuracy -> ' + str(accuracy) + '%'
    return length, indirect, direct, outage, accuracy

def print_files(train_files, test_files):
    print 'Train files:'
    for x in train_files:
        print x

    print 'Test files:'
    for x in test_files:
        print x

def calc_prediction(y_pred, y_test, num_classes):
    y_csv = pd.DataFrame()
    if num_classes > 2:
        for i in range(0, num_classes):
            y_test_list = map(lambda x: x[i], y_test)
            y_csv['y_test_'+str(i)] = pd.Series(list(y_test_list))

        for i in range(0, num_classes):
            y_pred_list = map(lambda x: x[i], y_pred)
            y_csv['y_pred_'+str(i)] = pd.Series(list(y_pred_list))
    else:
        y_pred_list = map(lambda x: x, y_pred)
        y_test_list = map(lambda x: x, y_test)
        y_csv['y_pred'] = pd.Series(list(y_pred_list))
        y_csv['y_test'] = pd.Series(list(y_test_list))

    return y_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--cparam', help='Number of epochs for all sequences', required=True)
    parser.add_argument('-k','--kernel', help='Kernel type - rbf, linear, poly, sigmoid', required=True)
    parser.add_argument('-f','--function', help='Classifier type - 1) SVC, 2) NuSVC, 3) LinearSVC', required=True)
    parser.add_argument('-t','--test', help='Test datasets (might be a comma-separated list)', required=True)
    parser.add_argument('-i','--ignore', help='List of datasets that must be ignored', required=False)
    parser.add_argument('-s','--steps', help='Number of steps to consider', required=True)
    args = vars(parser.parse_args())

    cparam = float(args['cparam'])
    function = int(args['function'])
    max_steps = int(args['steps'])
    test_events = args['test'].split(',')

    if args['kernel'] is not None:
        kernel = args['kernel']
    else:
        kernel = 'rbf'

    if args['ignore'] is not None:
        ignored_events = args['ignore'].split(',')
        ignored_events += test_events
    else:
        ignored_events = test_events

    train_files = get_train_datasets(ignored_events, multi = True, anomaly = True)
    test_files = get_test_datasets(test_events, multi = True, anomaly = True)

    # print_files(train_files, test_files)
    train_vals = []
    for file in train_files:
        x_val, y_val = csv_to_xy(file, 2, 0)
        train_vals.append(((x_val, y_val), file))

    test_vals = []
    for file in test_files:
        x_val, y_val = csv_to_xy(file, 2, 0)
        test_vals.append(((x_val, y_val), file))

    print '####TRAINING'

    l = []
    # x_total, y_total = (train_vals[0][0][0], train_vals[0][0][1])
    x_total, y_total = (train_vals[0][0][0][:max_steps, :], train_vals[0][0][1][:max_steps, :])
    for train_samples in train_vals[1:]:
        filename = train_samples[1]
        x_train, y_train = (train_samples[0][0][:max_steps,:], train_samples[0][1][:max_steps,:])
        x_total = np.append(x_total, x_train, axis=0)
        y_total = np.append(y_total, y_train, axis=0)

    xy_total = np.concatenate((x_total, y_total), axis=1)

    np.random.shuffle(xy_total)
    y_total = xy_total[:,-1:]

    #Compute weights in order to cope with unbalanced datasets
    sample_weights = compute_weights(xy_total)
    if function == 1:
        classif = SVC(kernel=kernel,C=cparam,random_state=0)
    elif function == 2:
        classif = NuSVC(kernel=kernel,random_state=0)
    elif function == 3:
        classif = LinearSVC(C=cparam,random_state=0)
    classif.fit(x_total, y_total, sample_weight=sample_weights)
    # classif = OneVsRestClassifier(estimator=SVC(gamma=0.001,random_state=0))
    # classif.fit(x_total, y_total)

    df = pd.DataFrame()

    print( '####VALIDATION')
    acc_total = 0
    for test_samples in test_vals:
        test_file = test_samples[1].split('/')[-1]
        x_test, y_test = test_samples[0]
        y_pred = classif.predict(x_test[0:max_steps,]).round()
        accuracy, precision, recall, f1 = calc_metrics(y_test[0:max_steps,], y_pred, multi=True)

        num_classes = 2
        y_csv = calc_prediction(y_pred, y_test, num_classes)
        y_csv.to_csv('results/y_pred_' + test_file + '.csv', quoting=3)

        length, indirect, direct, outage, accuracy = print_metrics(y_pred, accuracy, test_file)
        acc_total += accuracy
        df.set_value(test_file,'len', length)
        df.set_value(test_file,'indirect', indirect)
        df.set_value(test_file,'direct', direct)
        df.set_value(test_file,'outage', outage)
        df.set_value(test_file,'accuracy', accuracy)

    print 'TOTAL -> ' + str(np.round(acc_total/len(test_vals),2))
    model_name = 'svm_results_' + args['test'].replace(',','-')
    df.to_csv('results/'+model_name+'.csv', sep=',')

if __name__ == "__main__":
    main()
