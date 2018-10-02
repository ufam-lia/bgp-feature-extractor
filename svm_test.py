from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

from bgpanomalies import *
from lstm_detection import *

test_events = ['slammer','nimda','japan-earthquake','aws-leak']
train_files = get_train_datasets(test_events, multi = True, anomaly = True)
test_files = get_test_datasets(test_events, multi = True, anomaly = True)

for x in train_files:
    print x

for x in test_files:
    print x

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
x_total, y_total = (train_vals[0][0][0], train_vals[0][0][1])
for train_samples in train_vals[1:]:
    filename = train_samples[1]
    x_train, y_train = (train_samples[0][0], train_samples[0][1])
    x_total = np.append(x_total, x_train, axis=0)
    y_total = np.append(y_total, y_train, axis=0)
    print filename.split('multi_')[1] + '->' + str(y_train.shape[0])

classif = SVC(gamma=0.001,random_state=0)

for i in range(0,50):
    xy_total = np.concatenate((x_total, y_total), axis=1)
    np.random.shuffle(xy_total)
    y_total = xy_total[:,-1:]

    y_weights = to_categorical(y_total, num_classes=4)
    y_weights = y_weights.reshape(-1, 4)
    y_classes = pd.DataFrame(y_weights).idxmax(1, skipna=False)
    label_encoder = LabelEncoder()
    label_encoder.fit(list(y_classes))
    y_integers = label_encoder.transform(list(y_classes))
    sample_weights = compute_sample_weight('balanced', y_integers)

    np.savetxt('s1.csv',y_total,delimiter=',')
    np.savetxt('s2.csv',sample_weights,delimiter=',')

    # classif = OneVsRestClassifier(estimator=SVC(gamma=0.001,random_state=0))
    classif.fit(x_total, y_total, sample_weight=sample_weights)
# classif.fit(x_total, y_total)

for test_samples in test_vals:
    test_file = test_samples[1]
    x_test, y_test = test_samples[0]
    y_pred = classif.predict(x_test).round()

    accuracy, precision, recall, f1 = calc_metrics(y_test, y_pred, multi=True)
    # print_metrics(precision, recall, f1, test_file)

print( '####VALIDATION')
for test_samples in test_vals:
    test_file = test_samples[1].split('/')[-1]
    x_test, y_test = test_samples[0]
    y_pred = classif.predict(x_test).round()
    print y_pred
    accuracy, precision, recall, f1 = calc_metrics(y_test, y_pred, multi=True)
    print accuracy
    # df = save_metrics(accuracy, precision, recall, f1, test_file, df)

    # model_name = 'test_' + test_file + '_' + str(epochs) + 'x' + str(inner_epochs)+'x'+str(lag)
    # print model_name

    num_classes = 2

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


    # for i in range(0, num_classes):
    #     y_test_list = map(lambda x: x[i], y_test)
    #     y_csv['y_test_'+str(i)] = pd.Series(list(y_test_list))
    #
    # for i in range(0, num_classes):
    #     y_pred_list = map(lambda x: x[i], y_pred)
    #     y_csv['y_pred_'+str(i)] = pd.Series(list(y_pred_list))

    print test_file
    y_csv.to_csv('results/y_pred_' + test_file + '.csv', quoting=3)

# model_name = 'test_' + args['test'].replace(',','-') + '_' + str(epochs) + 'x' + str(inner_epochs)+'x'+str(lag)
# df.to_csv('results/results_'+model_name+'.csv', sep=',')
# classif.save('models/'+model_name + '.h5')
# print 'Results saved: results_'+model_name+'.csv'

# classif.fit(X, y)
# print classif.predict(X)
