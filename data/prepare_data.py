from read_functions import *
from create_vocabulary import *
from util import get_pkl_path

data_pkl_file = "data_final.pkl"

data = process_all()
statistics(data)
#data.to_pickle('data_final.pkl')
#data.to_pickle('../data/data_final.pkl')
data.to_pickle(get_pkl_path(data_pkl_file))
data.head()

#path='data_final.pkl'
#path='../data/data_final.pkl'
path = get_pkl_path(data_pkl_file)
df = pd.read_pickle(path)
diag_voc, med_voc, pro_voc = create_str_token_mapping(df)
records = create_patient_record(df, diag_voc, med_voc, pro_voc)
len(diag_voc.idx2word), len(med_voc.idx2word), len(pro_voc.idx2word)
