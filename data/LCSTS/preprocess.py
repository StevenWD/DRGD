import xmltodict
import numpy as np
import json
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from copy import deepcopy

def build_dict(content_list, base_dir):
    print('----build character dictionary')
    char_dict = dict()
    char_dict['<PAD>'] = len(char_dict)
    char_dict['<START>'] = len(char_dict)
    char_dict['<EOS>'] = len(char_dict)

    source_char_dict = deepcopy(char_dict)
    target_char_dict = deepcopy(char_dict)

    char_num = dict()
    for content in content_list:
        chars = [c for c in content]
        for c in chars:
            char_num[c] = char_num.get(c, 0) + 1
    
    for content in content_list:
        chars = [c for c in content]
        for c in chars:
            if c not in source_char_dict:
                source_char_dict[c] = len(source_char_dict)
                if char_num[c] > 100:
                    target_char_dict[c] = len(target_char_dict)

    json.dump(source_char_dict, open(base_dir+'data/LCSTS/source_char_dict.json', 'w'), indent=4, ensure_ascii=False)
    json.dump(target_char_dict, open(base_dir+'data/LCSTS/target_char_dict.json', 'w'), indent=4, ensure_ascii=False)
    return source_char_dict, target_char_dict

def build_emd(char_dict, emd_dim, base_dir, mode):
    print('----build embedding matrix')
    emd_weight = np.random.randn(len(char_dict), emd_dim).astype(np.float32)
    np.save(base_dir+'data/LCSTS/'+mode+'_emd_weight.npy', emd_weight)
    embedding_filepath = 'data/LCSTS/'+mode+'_emd_weight.npy'
    return embedding_filepath

def preprocess(fname, mode, config):
    print('Process file {} in mode {}'.format(fname, mode))
    f = open(fname, 'r')
    data = xmltodict.parse(f.read())
    summary_list = list()
    text_list = list()
    for item in data['data']['doc']:
        if mode != 'train' and int(item['human_label']) < 3:
            continue
        summary_list.append(item['summary'])
        text_list.append(item['short_text'])

    if mode == 'train':
        source_char_dict, target_char_dict = build_dict(text_list, config.base_dir)
        config.update_config('source_word_num', len(source_char_dict))
        config.update_config('target_word_num', len(target_char_dict))

        embedding_filepath = build_emd(source_char_dict, config.config['emd_dim'], config.base_dir, 'source')
        config.update_config('source_embedding_filepath', embedding_filepath)
        embedding_filepath = build_emd(target_char_dict, config.config['emd_dim'], config.base_dir, 'target')
        config.update_config('target_embedding_filepath', embedding_filepath)
        config.save_config()
    else:
        source_char_dict = json.load(open(config.base_dir+'data/LCSTS/source_char_dict.json', 'r'))
        target_char_dict = json.load(open(config.base_dir+'data/LCSTS/target_char_dict.json', 'r'))

   
    print('----converting data')
    source = list()
    source_length = list()
    target = list()
    target_length = list()
    for text in text_list:
        feature = list()
        for c in [tc for tc in text]:
            if c in source_char_dict:
                feature.append(source_char_dict[c])

        feature.append(source_char_dict['<EOS>'])
        source_length.append(len(feature))
        source.append(feature)

    source = pad_sequences(source,
            maxlen=config.config['source_max_seq_length'],
            dtype='int32',
            padding='post',
            truncating='post',
            value=source_char_dict['<PAD>'])

    for summary in summary_list:
        feature = list()
        for c in [tc for tc in summary]:
            if c in target_char_dict:
                feature.append(target_char_dict[c])

        feature.append(target_char_dict['<EOS>'])
        target_length.append(len(feature))
        target.append(feature)

    target = pad_sequences(target,
            maxlen=config.config['target_max_seq_length'],
            dtype='int32',
            padding='post',
            truncating='post',
            value=target_char_dict['<PAD>'])

    np.save(config.base_dir+'data/LCSTS/'+mode+'_source.npy', source)
    np.save(config.base_dir+'data/LCSTS/'+mode+'_target.npy', target)
    np.save(config.base_dir+'data/LCSTS/'+mode+'_source_length.npy', source_length)
    np.save(config.base_dir+'data/LCSTS/'+mode+'_target_length.npy', target_length)
