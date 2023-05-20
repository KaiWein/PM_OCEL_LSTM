import pandas as pd

act_for_filtered_order = ['place order','confirm order','payment reminder','pay order']
act_for_filtered_item = ['place order','confirm order', 'item out of stock', 'reorder item', 'pick item' , 'create package', 'send package', 'failed delivery', 'package delivered']
act_for_filtered_pack = ['create package', 'send package', 'failed delivery', 'package delivered']

## preparing the cel
def prepare_flat_ocel(fn, flatten_on, flattening = True,filter = None, printen_flat = False):
    """gets an filename from the source data and prepare it. transform columns. one-hot-encode activities and customer.
    This is adjustet for only the running-example OCEL. 
    With this we can also directly generate a filtered version(based on activitys)"""
    OCEL = pd.read_csv('../sourcedata/%s.csv' % fn)
    OCEL['ocel:timestamp']=  pd.to_datetime(OCEL['ocel:timestamp'])
    col_transform = {"ocel:eid": "Event_ID", "ocel:timestamp": "Timestamp", "ocel:activity": "Activity", "ocel:type:items": "Items",
                    "ocel:type:products": "Products", "ocel:type:customers":"Customers", "ocel:type:orders":"Orders",
                    "ocel:type:packages":"Packages"}
    OCEL = OCEL.rename(columns=col_transform).reindex(columns=[ 'Event_ID', 'Activity','Timestamp', 'weight', 'price', 'Items','Products', 'Customers', 'Orders', 'Packages'])
    OCEL['Items']=  OCEL['Items'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    OCEL['Products']=  OCEL['Products'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    OCEL['Customers']=  OCEL['Customers'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    OCEL['Orders']=  OCEL['Orders'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    OCEL['Packages']=  OCEL['Packages'].fillna('')
    OCEL['Packages']=  OCEL['Packages'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x]).apply(lambda x: x[-1])
    OCEL['Customers']=  OCEL['Customers'].apply(lambda x: x[-1])
    if filter is not None:
        OCEL = OCEL[OCEL['Activity'].isin(filter)]

    # Perform label encoding with letters and create a dictionary
    act_uni = OCEL['Activity'].unique()
    act_dict = {label: chr(ord('A') + i) for i, label in enumerate(act_uni)}
    OCEL['Activity'] = OCEL['Activity'].map(act_dict)

    cust_uni = OCEL['Customers'].unique()
    cust_dict = {label: chr(ord('a') + i) for i, label in enumerate(cust_uni)}
    OCEL['Customers'] = OCEL['Customers'].map(cust_dict)

    # pack_uni = ocel['Packages'].unique()
    # pack_dict = {label: chr(ord('a') + i) for i, label in enumerate(pack_uni)}
    # ocel['Packages'] = ocel['Packages'].map(pack_dict)
    """Flattens the OCEL log but do not remove any columns"""
    if flattening:
        OCEL = OCEL.explode(flatten_on).reset_index(drop=True)
        if printen_flat:
            print(f'size of normal OCEL:{OCEL.size}')
            pd.display(OCEL[OCEL['Event_ID']== 594])
            print(f'size of flattend OCEL:{OCEL.size}')
            pd.display(OCEL[OCEL['Event_ID']== 594])
            pd.display(OCEL.head())
    OCEL = OCEL.sort_values([flatten_on,'Timestamp'])
    return OCEL, act_dict, cust_dict #, pack_dict

#generating the csv files for the imput

def gen_enriched_single_plus_csv(OCEL,flatted_by, drops_col,csvname, printen = False):
    """This generates the enriched and the single logs for an flattend OCEL"""
    OCEL = OCEL.rename(columns={flatted_by:'Case_ID'})
    enriched_log = OCEL.set_index('Case_ID').reset_index().drop(columns=drops_col).sort_values(['Case_ID','Timestamp'])
    enriched_log['Amount_Items'] = [len(t) for t in enriched_log['Items']]
    enriched_log.to_csv(f'../data/{csvname}_enriched.csv')
    single_log = OCEL.set_index('Case_ID').reset_index()[['Case_ID','Activity','Timestamp']].sort_values(['Case_ID','Timestamp'])
    single_log.to_csv(f'../data/{csvname}_single.csv')
    if printen:
        pd.display(enriched_log[120:140])
        pd.display(single_log[120:140])
    return enriched_log, single_log

def gen_features(OCEL, add_last_case=False, columns_to_encode = ['Activity', 'Customers']):
    OCEL = time_features(OCEL=OCEL)
    # ad the ! as an indicater for the end of trace
    if ('!' not in list(OCEL['Activity'].unique())):
        columns = OCEL.columns
        # Create new rows with 'Case_ID' and 'Activity' columns filled with specific values
        new_rows = pd.DataFrame({'Case_ID': OCEL['Case_ID'].unique(), 'Activity': '!'})
        # Fill other columns with 0
        new_rows = new_rows.reindex(columns=columns, fill_value=0)
        if 'Packages' in OCEL.columns:
            new_rows['Packages'] = ""
        # Get the maximum timestamp for each Case_ID and add 1 second to move them at the end.
        max_timestamps = OCEL.groupby('Case_ID')['Timestamp'].max() + pd.Timedelta(seconds=1)
        new_rows['Timestamp'] = new_rows['Case_ID'].map(max_timestamps)
        # Concatenate the new rows with the original DataFrame, sort by 'Case_ID' and 'Timestamp'
        OCEL = pd.concat([OCEL, new_rows], ignore_index=True).sort_values(['Case_ID','Timestamp'])
        
    
    # gen features thath sayes if it is in package
    if 'Packages' in OCEL.columns:
        OCEL['In_Package'] = (OCEL['Packages'] != "").astype(int)

    
    ## one hot encoding
    OCEL = onehot_encode(OCEL, columns_to_encode)
    target_act_feat = list(filter(lambda k: 'Act_' in k, OCEL.columns))
    # Group by 'Case_ID' and shift the values of 'target_act_feat' by -1
    shifted_features = OCEL.groupby('Case_ID')[target_act_feat].shift(-1).fillna(0)

    # Add the shifted features to the 'enr_train' DataFrame
    OCEL = pd.concat([OCEL, shifted_features.add_prefix('Next_')], axis=1)
    if not add_last_case:
        OCEL = OCEL.loc[OCEL['Activity'] != '!']
    OCEL = OCEL.sort_values(['Case_ID','Timestamp'])
    return OCEL

def time_features(OCEL):
    """generates all time features"""
    # replaces Ptimeseqs (time difference between events) but care it is not grouped for the case_ID
    OCEL['Time_Diff'] = OCEL.groupby('Case_ID')['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().astype(int)
    # replaces Ptimeseqs2 (times since the start of the case) but care it is not grouped for the case_ID
    OCEL['Time_Since_Start'] = (OCEL['Timestamp'] - OCEL.groupby('Case_ID')['Timestamp'].transform('first')).dt.total_seconds().astype(int)
    # replaces Ptimeseqs3 (time since the midnight) but care it is not grouped for the case_ID
    OCEL['Time_Since_Midnight'] = (OCEL['Timestamp'] - OCEL['Timestamp'].dt.normalize()).dt.total_seconds().astype(int)
    # replaces Ptimeseqs4 (just weekday) but care it is not grouped for the case_ID
    OCEL['Weekday'] = OCEL['Timestamp'].dt.weekday
    # Calculate the remaining time for each case but care it is not grouped for the case_ID
    OCEL['Remaining_Time'] = OCEL.groupby('Case_ID')['Time_Since_Start'].transform(lambda x: x.max() - x)
    gen_feat = ['Time_Diff','Time_Since_Start', 'Time_Since_Midnight', 'Weekday', 'Remaining_Time']
    # Group by 'Case_ID' and shift the values of 'target_act_feat' by -1
    shifted_features = OCEL.groupby('Case_ID')[gen_feat].shift(-1).fillna(0)

    # Add the shifted features to the 'enr_train' DataFrame
    OCEL = pd.concat([OCEL, shifted_features.add_prefix('Next_')], axis=1)
    return OCEL

def onehot_encode(OCEL, columns_to_encode):
    """ Does the one hot encoding"""
    # Check if one-hot encoded columns already exist in DataFrame
    existing_columns = [col for col in OCEL.columns if col.startswith('Act_') or col.startswith('Cust_')]
    # Perform one-hot encoding only for new columns
    if len(existing_columns) == 0:
        # One-hot encode the activities
        if 'Activity' in columns_to_encode:
            activity_encoded = pd.get_dummies(OCEL['Activity'], dtype=int, prefix='Act')
            OCEL = pd.concat([OCEL, activity_encoded], axis=1)

        # One-hot encode the customers
        if 'Customers' in columns_to_encode:
            customer_encoded = pd.get_dummies(OCEL['Customers'], dtype=int, prefix='Cust')
            OCEL = pd.concat([OCEL, customer_encoded], axis=1)
    return OCEL

def gen_traces_and_maxlength_of_trace(OCEL):

    activity_sequences = OCEL.groupby('Case_ID')['Activity'].apply(lambda x: ''.join(x)).reset_index()
    maxlen = max([len(x) for x in activity_sequences['Activity']]) +1
    # maxlen = max(map(lambda x: len(x), activity_sequences))
    return activity_sequences, maxlen