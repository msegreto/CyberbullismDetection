import pandas as pd

def get_itemsets(frequent_itemsets, mode='closed'):
    
    assert mode in ['closed', 'maximal'], "You must choose between 'closed' or 'maximal' as mode value"
    
    selected = []
    for i, row_i in frequent_itemsets.iterrows():
        keep = True
        for j, row_j in frequent_itemsets.iterrows():
            if i == j:
                continue
            if row_i['itemsets'].issubset(row_j['itemsets']):
                if mode == 'closed' and row_i['support'] == row_j['support']:
                    keep = False
                    break
                if mode == 'maximal':
                    keep = False
                    break
        if keep:
            selected.append({'itemsets': row_i['itemsets'], 'support': row_i['support']})
    
    return pd.DataFrame(selected)