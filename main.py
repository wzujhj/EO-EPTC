import pickle
import random
import os
from transformer import *
from args import *
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def data_construct(streams):
    dataset = streams
    dataset_keys = list(dataset.keys())
    data = [dataset[key] for key in dataset_keys]
    return data

def Sequence_Feature_Reassembly_and_Segmentation(stream):
    local_length_sequence = []
    proxy_length_sequence = []

    if stream['proxy_type'] == 'sst':
        local_limit = 8158
    elif stream['proxy_type'] == 'trtt':
        local_limit = 8192
    elif stream['proxy_type'] == 'vmtt':
        local_limit = 8174
    elif stream['proxy_type'] == 'vltt':
        local_limit = 8192

    c2s_cnt = 0
    s2c_cnt = 0
    for packet in stream['local_length_seq']:
        if int(packet[1]) > 0:
            if packet[0] == 'c2s':
                if packet[4] == '1':
                    c2s_cnt += int(packet[1])
                    local_length_sequence.append(-1 * c2s_cnt)
                    c2s_cnt = 0
                else:
                    c2s_cnt += int(packet[1])
            else:
                if packet[4] == '1':
                    s2c_cnt += int(packet[1])
                    local_length_sequence.append(s2c_cnt)
                    s2c_cnt = 0
                else:
                    s2c_cnt += int(packet[1])
    local_length_sequence = Sequence_Feature_Segmentation(local_length_sequence, local_limit)

    if stream['proxy_type'] == 'sst':
        proxy_limit = 8192

        c2s_cnt = 0
        s2c_cnt = 0
        for packet in stream['proxy_length_seq']:
            if int(packet[1]) > 0:
                if packet[0] == 'c2s':
                    if packet[4] == '1':
                        c2s_cnt += int(packet[1])
                        proxy_length_sequence.append(-1 * c2s_cnt)
                        c2s_cnt = 0
                    else:
                        c2s_cnt += int(packet[1])
                else:
                    if packet[4] == '1':
                        s2c_cnt += int(packet[1])
                        proxy_length_sequence.append(s2c_cnt)
                        s2c_cnt = 0
                    else:
                        s2c_cnt += int(packet[1])
        proxy_length_sequence = Sequence_Feature_Segmentation(proxy_length_sequence, proxy_limit)

    else:
        c2s_cnt = 0
        s2c_cnt = 0
        for packet in stream['proxy_length_seq']:
            if packet[1] == '23':  # tls
                if packet[0] == 'c2s':
                    c2s_cnt += int(packet[2])
                    proxy_length_sequence.append(-1 * (int(packet[2])))
                    c2s_cnt = 0
                else:
                    s2c_cnt += int(packet[2])
                    proxy_length_sequence.append(int(packet[2]))
                    s2c_cnt = 0

    return local_length_sequence, proxy_length_sequence


def Sequence_Feature_Segmentation(seq_a, limit):
    seq_b = Split_Sequence(seq_a, limit)
    seq_c = [-x for x in seq_b]
    seq_d = Split_Sequence(seq_c, limit)
    seq_e = [-x for x in seq_d]
    return seq_e


def Split_Sequence(arr,limit):
    new_arr = []
    count = 0
    for i in range(len(arr)):
        if arr[i] > limit:
            count += arr[i]
        else:
            if count > 0:
                remainder = count % limit
                quotient = count // limit
                for j in range(quotient):
                    new_arr.append(limit)
                if remainder != 0:
                    new_arr.append(remainder)
                count = 0
            new_arr.append(arr[i])
    if count > 0:
        remainder = count % limit
        quotient = count // limit
        for j in range(quotient):
            new_arr.append(limit)
        if remainder != 0:
            new_arr.append(remainder)
    return new_arr


def Proxy_Handshak_Drop(stream, type, proxy_type):
    length_sequence = []

    if proxy_type == 'sst':
        local_c2s = 0
        local_s2c = 0
        proxy_c2s = 1
        proxy_s2c = 9

    elif proxy_type == 'trtt':
        local_c2s = 0
        local_s2c = 0
        proxy_c2s = 1
        proxy_s2c = 6


    elif proxy_type == 'vmtt':
        local_c2s = 0
        local_s2c = 0
        proxy_c2s = 3
        proxy_s2c = 7


    elif proxy_type == 'vltt':
        local_c2s = 0
        local_s2c = 0
        proxy_c2s = 1
        proxy_s2c = 6

    if type in ['proxy', 'translate']:
        for length in stream[type]:
            if length < 0:
                if proxy_c2s == 0:
                    length_sequence.append(length)
                else:
                    proxy_c2s -= 1
            else:
                if proxy_s2c == 0:
                    length_sequence.append(length)
                else:
                    proxy_s2c -= 1

    elif type in ['local']:
        for length in stream[type]:
            if length < 0:
                if local_c2s == 0:
                    length_sequence.append(length)
                else:
                    local_c2s -= 1
            else:
                if local_s2c == 0:
                    length_sequence.append(length)
                else:
                    local_s2c -= 1

    else:
        return TypeError
    if type in ['proxy', 'translate'] and sum([proxy_c2s, proxy_s2c]) != 0:
        return []
    return length_sequence


def code_vector(dataset):
    data_vector = []
    for stream in dataset:
        local_sq, proxy_sq = Sequence_Feature_Reassembly_and_Segmentation(stream)
        if len(local_sq) >= 16 and len(proxy_sq) >= 16:
            data_vector.append([local_sq, proxy_sq])
    return data_vector

def rf_data_construct(dataset, train_index, test_index):
    dataset_keys = list(dataset.keys())

    train_data_keys = [dataset_keys[idx] for idx in train_index]
    test_data_keys = [dataset_keys[idx] for idx in test_index]

    train_data = [dataset[key] for key in train_data_keys]
    test_data = [dataset[key] for key in test_data_keys]

    return train_data, test_data

def rf_vector(dataset, dataset_type, args, proxy_type):
    packet_limit = args.packet_limits
    data_vector = []
    if dataset_type == 'train':
        for stream in dataset:
            class_index = int(stream['site_index'])
            length_sequence = Proxy_Handshak_Drop(stream, 'translate', proxy_type)
            if len(length_sequence) < 16:
                continue
            local_length_seq = length_sequence[:packet_limit] + [0] * (packet_limit - len(length_sequence))
            data_vector.append([class_index] + local_length_seq)
    elif dataset_type == 'test':
        for stream in dataset:
            class_index = int(stream['site_index'])
            length_sequence = Proxy_Handshak_Drop(stream, 'proxy', proxy_type)
            if len(length_sequence) < 16:
                continue
            proxy_length_seq = length_sequence[:packet_limit] + [0] * (packet_limit - len(length_sequence))
            data_vector.append([class_index] + proxy_length_seq)
    else:
        raise KeyError
    return data_vector

def ML_RandomForest(train, test):
    train_vec = np.array(train)
    test_vec = np.array(test)

    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(train_vec[:, 1:], train_vec[:, :1])
    test_pred = rf.predict(test_vec[:, 1:])
    print(classification_report(y_pred=test_pred, y_true=test_vec[:, :1].flatten(), digits=4))


if __name__ == '__main__':
    # configure
    args = ArgumentParser()

    cache_dir = os.path.join('data', 'seq2seq')
    classify_cache_dir = os.path.join('data', 'seqclassify')
    res_cache_dir = os.path.join('data', 'result')
    dataset_file = 'vltt_seq2seq.cache.pk'
    classify_dataset_file = dataset_file.replace('seq2seq.cache.pk', 'seqclassify.cache.pk')

    dataset_file_path = os.path.join(cache_dir, dataset_file)
    proxy_type = dataset_file.split('_')[0]

    with open(dataset_file_path, 'rb') as fp:
        streams = pickle.load(fp)
    print('Load seq2seq Dataset Finished')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.set_device(args.gpu_id)
    random.seed(args.seed)

    # data construction
    train_data = data_construct(streams)

    # data preprocessing
    train_vector = code_vector(train_data)

    train_iter, src_vocab, tgt_vocab = load_data(train_vector, args, len(train_vector))

    encoder = TransformerEncoder(len(src_vocab), args)
    decoder = TransformerDecoder(len(tgt_vocab), args)

    net = EncoderDecoder(encoder, decoder)
    device = try_gpu(args.gpu_id)

    print('start training')
    train_seq2seq(net, train_iter, tgt_vocab, args, device)

    data_set = {}

    classify_dataset_file_path = os.path.join(classify_cache_dir, classify_dataset_file)
    print('dataset_file', classify_dataset_file)

    with open(classify_dataset_file_path, 'rb') as fp:
        streams_classify = pickle.load(fp)
    print('Load seqclassify Dataset Finished')

    total = len(streams_classify)
    pcount = 0
    percent_step = max(1, total // 10)
    last_reported_percent = 0

    for key in streams_classify:
        pcount += 1
        current_percent = (pcount / total) * 100

        if current_percent >= last_reported_percent + 10:
            print(f"Progress: {current_percent:.2f}%")
            last_reported_percent = current_percent


        stream = streams_classify[key]

        local_length_sequence, proxy_length_sequence = Sequence_Feature_Reassembly_and_Segmentation(stream)

        if len(local_length_sequence) >= 16 and len(proxy_length_sequence) >= 16:
            translate_length_sequence_unk = predict_seq2seq(net, local_length_sequence, src_vocab, tgt_vocab, args.num_steps, device)
            translate_length_sequence = unk2zero(translate_length_sequence_unk[0])
            data_set[key] = {}
            data_set[key]['site_class'] = stream['site_class']
            data_set[key]['site_index'] = stream['site_index']
            data_set[key]['local'] = local_length_sequence
            data_set[key]['proxy'] = proxy_length_sequence
            data_set[key]['translate'] = translate_length_sequence

    cache_path = os.path.join(res_cache_dir, proxy_type+'_classification' + '.cache.pk')
    print("Progress: 100%")
    with open(cache_path, 'wb') as fp:
        pickle.dump(data_set, fp)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    splits = kfold.split(data_set)
    for train_index, test_index in splits:
        train_data, test_data = rf_data_construct(data_set, train_index, test_index)
        print('Data Construction Finished')

        # data preprocessing
        train_vector = rf_vector(train_data, 'train', args, proxy_type)
        test_vector = rf_vector(test_data, 'test', args, proxy_type)

        ML_RandomForest(train_vector, test_vector)
