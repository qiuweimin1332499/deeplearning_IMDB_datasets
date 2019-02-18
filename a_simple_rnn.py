import numpy as np
time_steps=100
input_features=32
output_features=64

inputs=np.random.random((time_steps,input_features))
stat_t=np.zeros(output_features)

W=np.random.random((output_features,input_features))
U=np.random.random((output_features,output_features))
b=np.random.random(output_features)

successive_outputs=[]
for i in range(time_steps):
    output_t=np.tanh(np.dot(W,inputs[i])+np.dot(U,stat_t)+b)
    successive_outputs.append(output_t)
    stat_t=output_t
final_output_sequence=np.stack(successive_outputs,axis=0)
print(final_output_sequence)