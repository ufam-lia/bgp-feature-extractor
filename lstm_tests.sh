echo '@@@@@@ 10x1+0 @ Nimda'
time python lstm_detection.py 10 1  0
echo '@@@@@@ 10x1+1 @ Nimda'
time python lstm_detection.py 10 1  1
echo '@@@@@@ 10x1+3 @ Nimda'
time python lstm_detection.py 10 1  3

echo '@@@@@@ 1x10+0 @ Nimda'
time python lstm_detection.py 1 10  0
echo '@@@@@@ 1x10+1 @ Nimda'
time python lstm_detection.py 1 10  1
echo '@@@@@@ 1x10+3 @ Nimda'
time python lstm_detection.py 1 10  3

echo '@@@@@@ 30x1+0 @ Nimda'
time python lstm_detection.py 30 1  0
echo '@@@@@@ 30x1+1 @ Nimda'
time python lstm_detection.py 30 1  1
echo '@@@@@@ 30x1+3 @ Nimda'
time python lstm_detection.py 30 1  3

echo '@@@@@@ 1x30+0 @ Nimda'
time python lstm_detection.py 1 30  0
echo '@@@@@@ 1x30+1 @ Nimda'
time python lstm_detection.py 1 30  1
echo '@@@@@@ 1x30+3 @ Nimda'
time python lstm_detection.py 1 30  3

echo '@@@@@@ 10x10+0 @ Nimda'
time python lstm_detection.py 10 10  0
echo '@@@@@@ 10x10+1 @ Nimda'
time python lstm_detection.py 10 10  1
echo '@@@@@@ 10x10+3 @ Nimda'
time python lstm_detection.py 10 10  3

echo '@@@@@@ 30x5+0 @ Nimda'
time python lstm_detection.py 30 5  0
echo '@@@@@@ 30x5+1 @ Nimda'
time python lstm_detection.py 30 5  1
echo '@@@@@@ 30x5+3 @ Nimda'
time python lstm_detection.py 30 5  3

echo '@@@@@@ 30x30+0 @ Nimda'
time python lstm_detection.py 30 30  0
echo '@@@@@@ 30x30+1 @ Nimda'
time python lstm_detection.py 30 30  1
echo '@@@@@@ 30x30+3 @ Nimda'
time python lstm_detection.py 30 30  3
