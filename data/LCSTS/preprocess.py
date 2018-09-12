import xmltodict
import numpy as np
import json
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

def build_dict(summary_list, text_list, base_dir):
    print('----build character dictionary')
    char_dict = dict()
    char_dict['<UNK>'] = len(char_dict)
    char_dict['<START>'] = len(char_dict)
    char_dict['<END>'] = len(char_dict)
    char_dict['<PAD>'] = len(char_dict)

    for summary in summary_list:
        chars = [c for c in summary]
        for c in chars:
            if c not in char_dict:
                char_dict[c] = len(char_dict)

    for text in text_list:
        chars = [c for c in text]
        for c in chars:
            if c not in char_dict:
                char_dict[c] = len(char_dict)

    json.dump(char_dict, open(base_dir+'data/LCSTS/char_dict.json', 'w'), indent=4, ensure_ascii=False)
    return char_dict

def build_emd(char_num, emd_dim, base_dir):
    print('----build embedding matrix')
    emd_weight = np.random.randn(char_num, emd_dim).astype(np.float32)
    np.save(base_dir+'data/LCSTS/emd_weight.npy', emd_weight)
    embedding_filepath = 'data/LCSTS/emd_weight.npy'
    return embedding_filepath

def preprocess(fname, mode, config):
    print('Process file {} in mode {}'.format(fname, mode))
    f = open(fname, 'r')
    data = xmltodict.parse(f.read())
    summary_list = list()
    text_list = list()
    for item in data['data']['doc']:
        summary_list.append(item['summary'])
        text_list.append(item['short_text'])

    if mode == 'train':
        char_dict = build_dict(summary_list, text_list, config.base_dir)
        config.update_config('source_word_num', len(char_dict))
        config.update_config('target_word_num', len(char_dict))
        embedding_filepath = build_emd(len(char_dict), config.config['emd_dim'], config.base_dir)
        config.update_config('embedding_filepath', embedding_filepath)
        config.save_config()
    else:
        char_dict = json.load(open(config.base_dir+'data/LCSTS/char_dict.json', 'r'))
   
    print('----converting data')
    source = list()
    target = list()
    for text in text_list:
        feature = list()
        for c in [tc for tc in text]:
            feature.append(char_dict.get(c, char_dict['<UNK>']))
        feature.append(char_dict['<END>'])
        
        source.append(feature)

    source = pad_sequences(source,
            maxlen=config.config['source_max_seq_length'],
            dtype='int32',
            padding='post',
            truncating='post',
            value=char_dict['<PAD>'])

    for summary in summary_list:
        feature = list()
        for c in [tc for tc in summary]:
            feature.append(char_dict.get(c, char_dict['<UNK>']))
        feature.append(char_dict['<END>'])

        target.append(feature)

    target = pad_sequences(target,
            maxlen=config.config['target_max_seq_length'],
            dtype='int32',
            padding='post',
            truncating='post',
            value=char_dict['<PAD>'])

    np.save(config.base_dir+'data/LCSTS/'+mode+'_source.npy', source)
    np.save(config.base_dir+'data/LCSTS/'+mode+'_target.npy', target)
