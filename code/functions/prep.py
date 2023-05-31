import pandas as pd

def act_filter(on):
    if on == 'Orders':
        return ['place order','confirm order','payment reminder','pay order']
    elif on == 'Items':
        return ['place order','confirm order', 'item out of stock', 'reorder item', 'pick item' , 'create package', 'send package', 'failed delivery', 'package delivered']
    elif on == 'Packages':
        return ['create package', 'send package', 'failed delivery', 'package delivered']
    else:
        print('Wrong input must be Orders, Items or Packages')
def prep_ocel_complete(setting_inputs, csvname):
    ocel, act_dict, cust_dict = prepare_flat_ocel(setting_inputs)
    print(act_dict)
    print(cust_dict)

    # Create the enriched log and perform additional preprocessing, and save the single and enriched versions
    ocel = gen_enriched_single_plus_csv(OCEL=ocel, setting_input=setting_inputs, csvname=csvname)

    # Adding features
    ocel = generate_features(ocel, setting_inputs)

    # Folding the data
    return ocel, act_dict, cust_dict


def prepare_flat_ocel(setting_inputs, printen_flat=False):
    fl = None if setting_inputs['complete'] else act_filter(setting_inputs['flatten_by'])

    # Read the OCEL log from the source data file and transform columns
    OCEL = pd.read_csv('../sourcedata/running-example.csv')
    OCEL['ocel:timestamp'] = pd.to_datetime(OCEL['ocel:timestamp'])
    col_transform = {"ocel:eid": "Event_ID", "ocel:timestamp": "Timestamp", "ocel:activity": "Activity",
                     "ocel:type:items": "Items", "ocel:type:products": "Products",
                     "ocel:type:customers": "Customers", "ocel:type:orders": "Orders",
                     "ocel:type:packages": "Packages"}
    OCEL = OCEL.rename(columns=col_transform).reindex(columns=['Event_ID', 'Activity', 'Timestamp', 'weight',
                                                               'price', 'Items', 'Products', 'Customers', 'Orders',
                                                               'Packages'])
    OCEL['Items'] = OCEL['Items'].apply(lambda x: [t.replace("'", "") for t in x.strip('][').split(', ')])
    OCEL['Products'] = OCEL['Products'].apply(lambda x: [t.replace("'", "") for t in x.strip('][').split(', ')])
    OCEL['Customers'] = OCEL['Customers'].apply(lambda x: [t.replace("'", "") for t in x.strip('][').split(', ')])
    OCEL['Orders'] = OCEL['Orders'].apply(lambda x: [t.replace("'", "") for t in x.strip('][').split(', ')])
    OCEL['Packages'] = OCEL['Packages'].fillna('').apply(lambda x: [t.replace("'", "") for t in x.strip('][').split(', ')])
    OCEL['Packages'] = OCEL['Packages'].apply(lambda x: x[-1])
    OCEL['Customers'] = OCEL['Customers'].apply(lambda x: x[-1])

    if fl is not None:
        OCEL = OCEL[OCEL['Activity'].isin(fl)]

    # Perform label encoding with letters and create a dictionary
    act_uni = OCEL['Activity'].unique()
    act_dict = {label: chr(ord('A') + i) for i, label in enumerate(act_uni)}
    OCEL['Activity'] = OCEL['Activity'].map(act_dict)

    cust_uni = OCEL['Customers'].unique()
    cust_dict = {label: chr(ord('a') + i) for i, label in enumerate(cust_uni)}
    OCEL['Customers'] = OCEL['Customers'].map(cust_dict)

    # Flatten the OCEL log but do not remove any columns
    OCEL = OCEL.explode(setting_inputs['flatten_by']).reset_index(drop=True)
    if printen_flat:
        print(f"Size of normal OCEL: {OCEL.size}")
        pd.display(OCEL[OCEL['Event_ID'] == 594])
        print(f"Size of flattened OCEL: {OCEL.size}")
        pd.display(OCEL[OCEL['Event_ID'] == 594])
        pd.display(OCEL.head())

    OCEL = OCEL.sort_values([setting_inputs['flatten_by'], 'Timestamp'])
    return OCEL, act_dict, cust_dict




def gen_enriched_single_plus_csv(OCEL, setting_input, csvname, drops_col=["weight", "price", "Event_ID", 'Products'], printen=False):
    """This generates the enriched and the single logs for a flattened OCEL"""
    OCEL = OCEL.rename(columns={setting_input['flatten_by']: 'Case_ID'})
    OCEL.dropna(subset=['Case_ID'], inplace=True)
    OCEL = OCEL[OCEL['Case_ID'] != '']

    enriched_log = OCEL.set_index('Case_ID').reset_index().drop(columns=drops_col).sort_values(['Case_ID', 'Timestamp'])
    enriched_log.to_csv(f'../data/{csvname}_enriched.csv')

    single_log = OCEL.set_index('Case_ID').reset_index()[['Case_ID', 'Activity', 'Timestamp']].sort_values(['Case_ID', 'Timestamp'])
    single_log.to_csv(f'../data/{csvname}_single.csv')

    if printen:
        pd.display(enriched_log[120:140])
        pd.display(single_log[120:140])

    if not setting_input['single_log']:
        return enriched_log
    else:
        return single_log


def generate_features(OCEL, setting_input, add_last_case=False, columns_to_encode=['Activity', 'Customers']):
    single = setting_input['single_log']
    if single:
        columns_to_encode = ['Activity']

    if 'Packages' in OCEL.columns:
        OCEL['In_Package'] = (OCEL['Packages'] != "").astype(int)

    if 'Items' in OCEL.columns:
        OCEL['Amount_Items'] = OCEL['Items'].apply(len)

    if 'Orders' in OCEL.columns:
        OCEL['Amount_Orders'] = OCEL['Orders'].apply(len)

    OCEL = time_features(OCEL=OCEL)

    if '!' not in OCEL['Activity'].unique():
        columns = OCEL.columns
        new_rows = pd.DataFrame({'Case_ID': OCEL['Case_ID'].unique(), 'Activity': '!'})
        new_rows = new_rows.reindex(columns=columns, fill_value=0)

        if 'Packages' in OCEL.columns:
            new_rows['Packages'] = ""

        max_timestamps = OCEL.groupby('Case_ID')['Timestamp'].max() + pd.Timedelta(seconds=1)
        new_rows['Timestamp'] = new_rows['Case_ID'].map(max_timestamps)

        OCEL = pd.concat([OCEL, new_rows], ignore_index=True).sort_values(['Case_ID', 'Timestamp'])

    OCEL = onehot_encode(OCEL, columns_to_encode)

    target_act_feat = [col for col in OCEL.columns if 'Act_' in col] + ['Activity']
    shifted_features = OCEL.groupby('Case_ID')[target_act_feat].shift(-1).fillna(0)
    OCEL = pd.concat([OCEL, shifted_features.add_prefix('Next_')], axis=1)

    if not add_last_case:
        OCEL = OCEL.loc[OCEL['Activity'] != '!']

    OCEL['Position'] = OCEL.groupby('Case_ID').cumcount() + 1
    OCEL['Trace_Len'] = OCEL.groupby('Case_ID')['Position'].transform('max')
    OCEL = OCEL.sort_values(['Case_ID', 'Timestamp'])

    return OCEL


def time_features(OCEL):
    """Generates all time features"""
    # Calculate time difference between events within each case
    OCEL['Time_Diff'] = OCEL.groupby('Case_ID')['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().astype(int)
    
    # Calculate time since the start of each case
    OCEL['Time_Since_Start'] = (OCEL['Timestamp'] - OCEL.groupby('Case_ID')['Timestamp'].transform('first')).dt.total_seconds().astype(int)
    
    # Calculate time since midnight for each event
    OCEL['Time_Since_Midnight'] = (OCEL['Timestamp'] - OCEL['Timestamp'].dt.normalize()).dt.total_seconds().astype(int)
    
    # Extract weekday from the timestamp
    OCEL['Weekday'] = OCEL['Timestamp'].dt.weekday
    
    # Calculate the remaining time for each case
    OCEL['Remaining_Time'] = OCEL.groupby('Case_ID')['Time_Since_Start'].transform(lambda x: x.max() - x)
    
    gen_feat = ['Time_Diff', 'Time_Since_Start', 'Time_Since_Midnight', 'Weekday', 'Remaining_Time']
    
    # Group by 'Case_ID' and shift the values of generated features by -1
    shifted_features = OCEL.groupby('Case_ID')[gen_feat].shift(-1).fillna(0)

    # Add the shifted features to the DataFrame
    OCEL = pd.concat([OCEL, shifted_features.add_prefix('Next_')], axis=1)
    
    return OCEL


def onehot_encode(OCEL, columns_to_encode):
    """Performs one-hot encoding on specified columns in the DataFrame"""
    existing_columns = len([col for col in OCEL.columns if col.startswith('Act_') or col.startswith('Cust_')]) == 0

    if existing_columns:
        if 'Activity' in columns_to_encode:
            activity_encoded = pd.get_dummies(OCEL['Activity'], dtype=int, prefix='Act')
            OCEL = pd.concat([OCEL, activity_encoded], axis=1)

        if 'Customers' in columns_to_encode:
            customer_encoded = pd.get_dummies(OCEL['Customers'], dtype=int, prefix='Cust')
            OCEL = pd.concat([OCEL, customer_encoded], axis=1)
            if "Cust_0" in OCEL.columns:
                OCEL = OCEL.drop("Cust_0", axis=1)

    return OCEL


def gen_traces_and_maxlength_of_trace(OCEL):
    activity_sequences = OCEL.groupby('Case_ID')['Activity'].apply(''.join).reset_index()
    maxlen = activity_sequences['Activity'].str.len().max() + 1 # +1 for the end of the trace
    return activity_sequences, maxlen
