import pandas as pd

## preparing the cel
def prepare_ocel(fn, filter = None):
    ocel = pd.read_csv('../sourcedata/%s.csv' % fn)
    ocel['ocel:timestamp']=  pd.to_datetime(ocel['ocel:timestamp'])
    col_transform = {"ocel:eid": "Event_ID", "ocel:timestamp": "Timestamp", "ocel:activity": "Activity", "ocel:type:items": "Items",
                    "ocel:type:products": "Products", "ocel:type:customers":"Customers", "ocel:type:orders":"Orders",
                    "ocel:type:packages":"Packages"}
    ocel = ocel.rename(columns=col_transform).reindex(columns=[ 'Event_ID', 'Activity','Timestamp', 'weight', 'price', 'Items','Products', 'Customers', 'Orders', 'Packages'])
    ocel['Items']=  ocel['Items'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    ocel['Products']=  ocel['Products'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    ocel['Customers']=  ocel['Customers'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    ocel['Orders']=  ocel['Orders'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x])
    ocel['Packages']=  ocel['Packages'].fillna('')
    ocel['Packages']=  ocel['Packages'].apply(lambda x: x.strip('][').split(', ')).apply(lambda x: [t.replace("'","") for t in x]).apply(lambda x: x[-1])
    ocel['Customers']=  ocel['Customers'].apply(lambda x: x[-1])
    if filter is not None:
        ocel = ocel[ocel['Activity'].isin(filter)]

    # Perform label encoding with letters and create a dictionary
    act_uni = ocel['Activity'].unique()
    act_dict = {label: chr(ord('A') + i) for i, label in enumerate(act_uni)}
    ocel['Activity'] = ocel['Activity'].map(act_dict)

    cust_uni = ocel['Customers'].unique()
    cust_dict = {label: chr(ord('a') + i) for i, label in enumerate(cust_uni)}
    ocel['Customers'] = ocel['Customers'].map(cust_dict)

    # pack_uni = ocel['Packages'].unique()
    # pack_dict = {label: chr(ord('a') + i) for i, label in enumerate(pack_uni)}
    # ocel['Packages'] = ocel['Packages'].map(pack_dict)

    return ocel, act_dict, cust_dict #, pack_dict

## flattening on order
def flatten(OCEL, on, printen = False):
    ocel_flat = OCEL.explode(on).reset_index(drop=True)
    if printen:
        print(f'size of normal OCEL:{OCEL.size}')
        pd.display(OCEL[OCEL['Event_ID']== 594])
        print(f'size of flattend OCEL:{ocel_flat.size}')
        pd.display(ocel_flat[ocel_flat['Event_ID']== 594])
        pd.display(ocel_flat.head())
    return ocel_flat

#generating the csv files for the imput

def gen_enriched_single_plus_csv(OCEL,flatted_by, drops_col,csvname, printen = False):
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

act_for_filtered_order = ['place order','confirm order','payment reminder','pay order']
act_for_filtered_item = ['place order','confirm order', 'item out of stock', 'reorder item', 'pick item' , 'create package', 'send package', 'failed delivery', 'package delivered']
act_for_filtered_pack = ['create package', 'send package', 'failed delivery', 'package delivered']
