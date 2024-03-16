import pandas as pd
import numpy as np

filename = "src/C_array_MF_smalldt.csv"

df = pd.read_csv(filename, sep="[")
arr = df.to_numpy()

string = arr[0,0]
print(len(string))
split_commas = string.split(",")

replace_0 = [item.replace('[', '') for item in split_commas]

replace_1 = [item.replace(']', '') for item in replace_0]

replace_1[0]= replace_1[0].replace("Vector{BigFloat}", "")

float_arr = [float(string) for string in replace_1]

float_arr = np.array(float_arr)

np.savetxt("src/C_arr_MF_fixed.csv", float_arr, delimiter=",")
