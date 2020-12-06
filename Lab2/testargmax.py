import numpy as np
q_values={}
q_values[(0, 0)]=10
q_values[(0, 1)]=10
q_values[(0, 2)]=2
q_values[(0, 3)]=3

relevant_qs = [q_values[(0, action)] for action in range(0, 4)]

print([np.argmax(relevant_qs)])
