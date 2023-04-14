import pandas as pd
import numpy as np
from collections import defaultdict
import dill

# atc -> cid
#ddi_file = 'drug-DDI.csv'
#cid_atc = 'drug-atc.csv'
#voc_file = 'voc_final.pkl'
#data_path = 'records_final.pkl'
data_path_prefix = "../data/prepare_data/data/"
pkl_path_prefix = "../data/"
ddi_file = '{}drug-DDI.csv'.format(data_path_prefix)
cid_atc = '{}drug-atc.csv'.format(data_path_prefix)
voc_file = '{}voc_final.pkl'.format(pkl_path_prefix)
data_path = '{}records_final.pkl'.format(pkl_path_prefix)

TOPK = 40 # topk drug-drug interaction

records =  dill.load(open(data_path, 'rb'))
cid2atc_dic = defaultdict(set)
med_voc = dill.load(open(voc_file, 'rb'))['med_voc']
med_voc_size = len(med_voc.idx2word)
med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
atc3_atc4_dic = defaultdict(set)
for item in med_unique_word:
    atc3_atc4_dic[item[:4]].add(item)

with open(cid_atc, 'r') as f:
    for line in f:
        line_ls = line[:-1].split(',')
        cid = line_ls[0]
        atcs = line_ls[1:]
        for atc in atcs:
            if len(atc3_atc4_dic[atc[:4]]) != 0:
                cid2atc_dic[cid].add(atc[:4])

# ddi load
ddi_df = pd.read_csv(ddi_file)
# fliter sever side effect
ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
ddi_most_pd = ddi_most_pd.iloc[-TOPK:,:]
# ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
ddi_df = fliter_ddi_df[['STITCH 1','STITCH 2']].drop_duplicates().reset_index(drop=True)

# weighted ehr adj
ehr_adj = np.zeros((med_voc_size, med_voc_size))
for patient in records:
    for adm in patient:
        med_set = adm[2]
        for i, med_i in enumerate(med_set):
            for j, med_j in enumerate(med_set):
                if j<=i:
                    continue
                ehr_adj[med_i, med_j] = 1
                ehr_adj[med_j, med_i] = 1
#dill.dump(ehr_adj, open('ehr_adj_final.pkl', 'wb'))
dill.dump(ehr_adj, open('../data/ehr_adj_final.pkl', 'wb'))


# ddi adj
ddi_adj = np.zeros((med_voc_size,med_voc_size))
for index, row in ddi_df.iterrows():
    # ddi
    cid1 = row['STITCH 1']
    cid2 = row['STITCH 2']

    # cid -> atc_level3
    for atc_i in cid2atc_dic[cid1]:
        for atc_j in cid2atc_dic[cid2]:

            # atc_level3 -> atc_level4
            for i in atc3_atc4_dic[atc_i]:
                for j in atc3_atc4_dic[atc_j]:
                    if med_voc.word2idx[i] != med_voc.word2idx[j]:
                        ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                        ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1


#dill.dump(ddi_adj, open('ddi_A_final.pkl', 'wb'))
dill.dump(ddi_adj, open('../data/ddi_A_final.pkl', 'wb'))
print('complete!')
