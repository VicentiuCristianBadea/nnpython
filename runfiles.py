import os, time

_batchVar = [1, 20, 50, 100]
_epochVar = [10, 20, 50, 100]
_learningRateVar = [0.1, 0.01, 0.001, 0.0001]

_theta1_R = 785
_theta1_C = [25, 50, 75, 100]
_theta2_R = [26, 51, 76, 101]
_theta2_C = [25, 50, 75, 100]
_theta3_R = [26, 51, 76, 101]
_theta3_C = 10
args = []

for batch in _batchVar:
	for epoch in _epochVar:
		for rate in _learningRateVar:
			for theta in range(4):
				_theta1_r = _theta1_R
				_theta1_c = _theta1_C[theta]
				_theta2_r = _theta2_R[theta]
				_theta2_c = _theta2_C[theta]
				_theta3_r = _theta3_R[theta]
				_theta3_c = _theta3_C
				args = [_theta1_r, _theta1_c, _theta2_r, _theta2_c, _theta3_r, _theta3_c, batch, epoch, rate]
				args = [str(x) for x in args]
				print(args)
				args_joined = " ".join(args)
				print(args_joined)
				os.system('python nnpython.py'+" "+args_joined)	
