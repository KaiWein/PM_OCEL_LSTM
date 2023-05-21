import numpy as np
from functions import prep

def generating_inputs(OCEL, num_of_features, max_trace_length, taf, act, divisor_next, divisor_since, divisor_remaining, single=False, custf=None, test=False, prefix_length=None):
    number_of_train_cases = len(OCEL)
    act_pos = len(act) 
    # Adjust max_trace_length if prefix_length is specified
    if prefix_length is not None:
        max_trace_length = prefix_length + 1
    else:
        prefix_length = max_trace_length + 1 
    

    # taf = target activity features
    X = np.zeros((number_of_train_cases, max_trace_length, num_of_features), dtype=np.float32)

    if not single and custf is not None:
        cust_pos = len(custf)
        onehot_offset = act_pos + cust_pos
        pos_Amount_Items = onehot_offset + 5
        pos_In_Package = onehot_offset + 6

        cust_values = OCEL[custf].values
        amount_items_values = OCEL['Amount_Items'].values
        in_package_values = OCEL['In_Package'].values

    else:
        onehot_offset = act_pos

    pos_Position = onehot_offset
    pos_Time_Diff = onehot_offset + 1
    pos_Time_Since_Start = onehot_offset + 2
    pos_Time_Since_Midnight = onehot_offset + 3
    pos_Weekday = onehot_offset + 4

    pos = OCEL['Position'].values
    act_values = OCEL[act].values[: :] 
    position_values = OCEL['Position'].values
    time_diff_values = OCEL['Time_Diff'].values / divisor_next
    time_start_values = OCEL['Time_Since_Start'].values  / divisor_since
    time_midnight_values = OCEL['Time_Since_Midnight'].values  / 86400
    weekday_values = OCEL['Weekday'].values  / 7

    if not single:
        for i in range(number_of_train_cases):
            posi = min(pos[i], prefix_length)
            if posi > max_trace_length:
                continue
            leftpad = max_trace_length - posi

            X[i, leftpad:, :act_pos] = act_values[i - posi + 1:i + 1, :]
            X[i, leftpad:, act_pos:act_pos + cust_pos] = cust_values[i - posi + 1:i + 1, :]
            X[i, leftpad:, pos_Position] = position_values[i - posi + 1:i + 1]
            X[i, leftpad:, pos_Time_Diff] = time_diff_values[i - posi + 1:i + 1]
            X[i, leftpad:, pos_Time_Since_Start] = time_start_values[i - posi + 1:i + 1]
            X[i, leftpad:, pos_Time_Since_Midnight] = time_midnight_values[i - posi + 1:i + 1]
            X[i, leftpad:, pos_Weekday] = weekday_values[i - posi + 1:i + 1]
            X[i, leftpad:, pos_Amount_Items] = amount_items_values[i - posi + 1:i + 1]
            X[i, leftpad:, pos_In_Package] = in_package_values[i - posi + 1:i + 1]

    elif single:
        for i in range(number_of_train_cases):
            posi = min(pos[i], prefix_length)
            if posi > max_trace_length:
                continue
            leftpad = max_trace_length - posi
            X[i, leftpad:, :act_pos] = act_values[i - posi + 1:i + 1, :]
            X[i, leftpad:, pos_Position] = position_values[i - posi + 1:i + 1]
            X[i, leftpad:, pos_Time_Diff] = time_diff_values[i - posi + 1:i + 1]
            X[i, leftpad:, pos_Time_Since_Start] = time_start_values[i - posi + 1:i + 1]
            X[i, leftpad:, pos_Time_Since_Midnight] = time_midnight_values[i - posi + 1:i + 1]
            X[i, leftpad:, pos_Weekday] = weekday_values[i - posi + 1:i + 1]

    y_a = OCEL.loc[:, taf].to_numpy(dtype=np.float32)
    y_t = OCEL['Next_Time_Diff'].to_numpy(dtype=np.float32) / divisor_next
    if test:
        y_tr = OCEL['Remaining_Time'].to_numpy(dtype=np.float32) / divisor_remaining
    else:
        y_tr = OCEL['Next_Remaining_Time'].to_numpy(dtype=np.float32) / divisor_remaining

    return X, y_a, y_t, y_tr
