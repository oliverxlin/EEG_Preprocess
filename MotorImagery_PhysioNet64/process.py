import json, mne, pyedflib
import matplotlib.pyplot as plt
import numpy as np


def get_data_and_label(raw_data):
    """
    input: raw_data: mne.io.read_raw_edf 的输出
    return: data->[trials, sig_len, electrodes]
            label->[trials]
    只返回完整的片段，不做任何处理(4.1s * 160hz)
    """
    events_from_annot, event_dict = mne.events_from_annotations(raw_data, verbose = 0)
    # 对应的信号和事件
    signals = raw_data.to_data_frame().to_numpy()
    events_from_annot = np.array(events_from_annot)

    sig_len = 4.1 * 160
    # 获取每一个event的起始点、终点
    events_segment = np.ones((events_from_annot.shape[0], 2))
    events_segment[:, 0] = events_from_annot[:, 0]
    events_segment[:, 1] = events_from_annot[:, 0] + sig_len
    events_segment = events_segment.astype(int)

    # 将信号按照event进行划分
    signal_seg = []
    for start, end in events_segment:
        signal_seg.append(signals[start :end, :])
    signal_seg = np.stack(signal_seg, axis=0)

    # 获取每个event的标签,T0T1T2对应(1, 2, 3)
    # 1是rest标签, 2和3对应T1和T2
    # 14次实验，4, 8, 12是(T1左拳T2右拳), 6, 10, 14(T1是双拳T2双脚)
    event_label = events_from_annot[:, 2]


    # 取出T1T2的数据
    idx = np.argwhere(event_label >= 2).squeeze()
    # 将标签处理成1-2
    label = event_label[idx] - 1
    data = signal_seg[idx]
    return data, label


def load_physionet_mi(sub = 10, num_classes = 4, verbose = 1):
    """
    input: sub-> subject id
           num_classes->2 or 4
           verbose->1 print, 0 no print
    return: data->[trials, sig_len, electrodes]
            label->[trials]
    只返回完整的片段，不做任何处理(4.1s * 160hz)
    """
    data_path = "S%03d/S%03dR%02d.edf"   # 存放数据的具体位置，需要改成自己数据存放的地方
    MI_RUNS_1 = [4, 8, 12]
    MI_RUNS_2 = [6, 10, 14]
    runs = 14

    if num_classes != 4:
        MI_RUNS_2 = [] 
    data, label = [], []
    for run in range(runs):
        if run + 1 in MI_RUNS_1 or run + 1 in MI_RUNS_2:
            raw_data = mne.io.read_raw_edf(data_path % (sub, sub, run + 1), preload=True, verbose = 0)
            data_temp, label_temp = get_data_and_label(raw_data)
            # 将分类处理成3-4(两组都是1-2，所以第二组需要+2处理成3-4))
            if run + 1 in MI_RUNS_2:
                label_temp += 2
            data.append(data_temp)
            label.append(label_temp)
    if verbose >= 1:
        print("受试者%d 的数据加载完毕" % (sub))
    return np.concatenate(data, axis=0), np.concatenate(label, axis=0)

if __name__ == "__main__":
    data, label = load_physionet_mi(sub=10, num_classes=4, verbose=0)
    print(data.shape, label.shape)
    # (90, 656, 65) (90,)