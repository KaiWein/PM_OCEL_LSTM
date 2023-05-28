import numpy as np
from functions import prep

def generating_inputs(OCEL, num_of_features, max_trace_length, taf, act, divisor_next, divisor_since, divisor_remaining, custf=None, test=False, prefix_length=0):
    pack_flag = 'In_Package' in OCEL.columns
    item_flag = 'Amount_Items' in OCEL.columns
    order_flag = 'Amount_Orders' in OCEL.columns
    trace_length = OCEL['Trace_Len'].values
    OCEL = OCEL[trace_length >= prefix_length].reset_index(drop= True)
    
    number_of_train_cases = len(OCEL)
    act_pos = len(act) 

    if prefix_length != 0:
        max_trace_length = prefix_length
    else:
        prefix_length = max_trace_length + 1
    X = np.zeros((number_of_train_cases, max_trace_length, num_of_features), dtype=np.float32)

    if custf is not None:
        cust_pos = len(custf)
        onehot_offset = act_pos + cust_pos
        cust_values = OCEL[custf].values
        i = 5
        if pack_flag:
            in_package_values = OCEL['In_Package'].values
            pos_In_Package = onehot_offset + i
            i = i +1
        if item_flag:
            amount_items_values = OCEL['Amount_Items'].values
            pos_Amount_Items = onehot_offset + i
            i = i +1
        if order_flag:
            amount_orders_values = OCEL['Amount_Orders'].values
            pos_orders_Items = onehot_offset + i
            i = i +1

    else:
        onehot_offset = act_pos

    pos_Position = onehot_offset
    pos_Time_Diff = onehot_offset + 1
    pos_Time_Since_Start = onehot_offset + 2
    pos_Time_Since_Midnight = onehot_offset + 3
    pos_Weekday = onehot_offset + 4

    pos = OCEL['Position'].values
    trace_length = OCEL['Trace_Len'].values
    act_values = OCEL[act].values[: :] 
    position_values = OCEL['Position'].values
    time_diff_values = OCEL['Time_Diff'].values / divisor_next
    time_start_values = OCEL['Time_Since_Start'].values  / divisor_since
    time_midnight_values = OCEL['Time_Since_Midnight'].values  / 86400
    weekday_values = OCEL['Weekday'].values  / 7

    for i in range(number_of_train_cases):
        posi = min(pos[i], prefix_length)
        leftpad = max_trace_length - posi

        X[i, leftpad:, :act_pos] = act_values[i - posi + 1:i + 1, :]
        if custf is not None:
            X[i, leftpad:, act_pos:act_pos + cust_pos] = cust_values[i - posi + 1:i + 1, :]
        X[i, leftpad:, pos_Position] = position_values[i - posi + 1:i + 1]
        X[i, leftpad:, pos_Time_Diff] = time_diff_values[i - posi + 1:i + 1]
        X[i, leftpad:, pos_Time_Since_Start] = time_start_values[i - posi + 1:i + 1]
        X[i, leftpad:, pos_Time_Since_Midnight] = time_midnight_values[i - posi + 1:i + 1]
        X[i, leftpad:, pos_Weekday] = weekday_values[i - posi + 1:i + 1]
        if pack_flag:
            X[i, leftpad:, pos_In_Package] = in_package_values[i - posi + 1:i + 1]
        if item_flag:
            X[i, leftpad:, pos_Amount_Items] = amount_items_values[i - posi + 1:i + 1]
        if order_flag:
            X[i, leftpad:, pos_orders_Items] = amount_orders_values[i - posi + 1:i + 1]

    y_a = OCEL.loc[:, taf].to_numpy(dtype=np.float32)
    y_t = OCEL['Next_Time_Diff'].to_numpy(dtype=np.float32) / divisor_next
    if test:
        y_tr = OCEL['Remaining_Time'].to_numpy(dtype=np.float32) / divisor_remaining
    else:
        y_tr = OCEL['Next_Remaining_Time'].to_numpy(dtype=np.float32) / divisor_remaining

    return X, y_a, y_t, y_tr
