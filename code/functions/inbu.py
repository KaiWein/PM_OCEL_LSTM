import numpy as np 


def generating_inputs(OCEL, taf, divisor_next, divisor_remaining):
    ## taf = target activity features
    y_a = OCEL.loc[:,taf].to_numpy(dtype=np.float32)
    y_t = OCEL['Next_Time_Diff'].to_numpy(dtype=np.float32)/ divisor_next
    y_tr = OCEL['Next_Remaining_Time'].to_numpy(dtype=np.float32)/ divisor_remaining
    return y_a, y_t, y_tr