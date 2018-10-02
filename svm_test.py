from __future__ import division
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

from bgpanomalies import *
from lstm_detection import *
from collections import Counter

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
    print test_file
    print 'len      -> '  + str(len(y_pred))
    print 'indirect -> '  + str(np.round(count[1]/len(y_pred)*100,2)) + '%'
    print 'direct   -> '  + str(np.round(count[2]/len(y_pred)*100,2)) + '%'
    print 'outage   -> '  + str(np.round(count[3]/len(y_pred)*100,2)) + '%'
    print '*** accuracy -> ' + str(np.round(accuracy*100,2)) + '%'

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
    parser.add_argument('-t','--test', help='Test datasets (might be a comma-separated list)', required=True)
    parser.add_argument('-i','--ignore', help='List of datasets that must be ignored', required=False)
    parser.add_argument('-s','--steps', help='Number of steps to consider', required=True)
    args = vars(parser.parse_args())

    max_steps = int(args['steps'])
    test_events = args['test'].split(',')

    if args.has_key('ignore'):
        ignored_events = args['ignore'].split(',')
        ignored_events += test_events
    else:
        ignored_events = test_events

    train_files = get_train_datasets(ignored_events, multi = True, anomaly = True)
    test_files = get_test_datasets(test_events, multi = True, anomaly = True)

    print_files(train_files, test_files)
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

    print type(train_vals)
    print type(train_vals[0])
    print type(train_vals[0][0])
    print type(train_vals[0][0][0])

    xy_total = np.concatenate((x_total, y_total), axis=1)
    np.random.shuffle(xy_total)
    y_total = xy_total[:,-1:]

    #Compute weights in order to cope with unbalanced datasets
    sample_weights = compute_weights(xy_total)
    print sample_weights
    print y_total

    print sample_weights.shape
    print x_total.shape
    print y_total.shape
    classif = SVC(gamma=0.001,random_state=0)
    # classif = OneVsRestClassifier(estimator=SVC(gamma=0.001,random_state=0))
    classif.fit(x_total, y_total, sample_weight=sample_weights)
    # classif.fit(x_total, y_total)

    print( '####VALIDATION')
    for test_samples in test_vals:
        test_file = test_samples[1].split('/')[-1]
        x_test, y_test = test_samples[0]
        y_pred = classif.predict(x_test[0:max_steps,]).round()
        accuracy, precision, recall, f1 = calc_metrics(y_test[0:max_steps,], y_pred, multi=True)

        num_classes = 2
        print_metrics(y_pred, accuracy, test_file)
        y_csv = calc_prediction(y_pred, y_test, num_classes)
        y_csv.to_csv('results/y_pred_' + test_file + '.csv', quoting=3)

if __name__ == "__main__":
    main()
