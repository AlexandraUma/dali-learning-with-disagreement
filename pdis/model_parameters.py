# things that should have been in config

glove_300d_filtered_pd = {'path':'/homes/anu30/in_common/embeddings/glove.840B.300d.txt.filtered', 'size':300}

glove_300d_2w_filtered_pd = {'path': '/homes/anu30/in_common/embeddings/glove_50_300_2.txt.filtered', 'size':300}

char_vocab_path = "/homes/anu30/in_common/embeddings/char_vocab.english.txt"

max_word_length = 30

lm_path = '/homes/anu30/in_common/embeddings/pd2.0_jmlr_elmo_cache.hdf5'

#train_path =  "../mace_pd_dndo/mace_train_dndo_raw.jsonlines"
#dev_path =  "../mace_pd_dndo/mace_dev_dndo_raw.jsonlines"
#test_path =  "../mace_pd_dndo/mace_test_dndo_raw.jsonlines"



train_path =  "../pd_is_data/combined_train_dndo_raw.jsonlines"
dev_path =  "../pd_is_data/combined_dev_dndo_raw.jsonlines"
test_path =  "../pd_is_data/combined_test_dndo_raw.jsonlines"


filter_widths = [3,4,5]

genres = {'un':0, 'wi':1, 'gu':2}

class_names = ['DN', 'DO']

num_classes = len(class_names)

num_users = 1767

