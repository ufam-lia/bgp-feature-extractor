: '
time python feature-extractor.py rrc03 3257 as-3561-filtering 1 &
time python feature-extractor.py rrc03 3333 as-3561-filtering 1 &
time python feature-extractor.py rrc03 286 as-3561-filtering 1

time python feature-extractor.py rrc05 12793 as9121 1 &
time python feature-extractor.py rrc05 13237 as9121 1 &
time python feature-extractor.py rrc05 1853 as9121 1

time python feature-extractor.py rrc01 13237 as-depeering 1 
time python feature-extractor.py rrc01 8342 as-depeering 1
time python feature-extractor.py rrc01 5511 as-depeering 1 
time python feature-extractor.py rrc01 16034 as-depeering 1
'

time python feature-extractor.py rrc03 3257 as-path-error 1
time python feature-extractor.py rrc03 3333 as-path-error 1
time python feature-extractor.py rrc03 286 as-path-error 1
time python feature-extractor.py rrc03 9057 as-path-error 1

:'
time python feature-extractor.py rrc04 15547 aws-leak 1
time python feature-extractor.py rrc04 25091 aws-leak 1
time python feature-extractor.py rrc04 34781 aws-leak 1

time python feature-extractor.py rrc04 513 code-red 1 &
time python feature-extractor.py rrc04 559 code-red 1 &
time python feature-extractor.py rrc04 6893 code-red 1

time python feature-extractor.py rrc04 513 malaysian-telecom 1
time python feature-extractor.py rrc04 34781 malaysian-telecom 1
time python feature-extractor.py rrc04 25091 malaysian-telecom 1
time python feature-extractor.py rrc04 20932 malaysian-telecom 1

time python feature-extractor.py rrc05 1853 moscow-blackout 1 &
time python feature-extractor.py rrc05 12793 moscow-blackout 1 &
time python feature-extractor.py rrc05 13237 moscow-blackout 1 &

time python feature-extractor.py rrc04 513 nimda 10 &
time python feature-extractor.py rrc04 559 nimda 10 &
time python feature-extractor.py rrc04 6893 nimda 10 &

time python feature-extractor.py rrc04 513 slammer 10 &
time python feature-extractor.py rrc04 559 slammer 10 &
time python feature-extractor.py rrc04 6893 slammer 10 &
'
