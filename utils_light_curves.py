import numpy as np
from tqdm import tqdm


def add_filtered_signals(x, y):
    mask = y != -99
    x_copy = np.copy(x[mask])
    seq_len = (x_copy[:, :, 2] != 0).sum(1)
    for i, xi in tqdm(enumerate(x_copy)):
        dt = xi[1:, 0] - xi[:-1, 0]
        filtered_signal = moving_average(xi[:seq_len[i], [0, 1]], dt, dt.max() / 10)
        x_copy[i, :seq_len[i], 1] = filtered_signal
    x = np.concatenate((x, x_copy), axis=0)
    y = np.concatenate((y, y[mask]), axis=0)
    print(x.shape, y.shape)
    return x, y


def moving_average(x, dt, t):
    y_values = []
    y = x[0, 1]
    t_i = 0    
    y_values.append(y)
    for i, xi in enumerate(x[1:]):
        dti = dt[i]
        y += (1 - np.exp(- dti / t)) * (xi[1] - y)
        y_values.append(y)
    return y_values


def process_self_adversarial(x, y, args):        
    # train auxiliary data
    index = np.arange(len(x))
    np.random.shuffle(index)
    x = x[index]
    y = y[index]

    aux_x = np.empty((0, x.shape[1], x.shape[2]))
    if args["TS"]:
        if args["shift_index"] == "random":
            aux_ts = time_shift_surrogates(x, 0, random_shift=True)
            aux_x = np.concatenate((aux_x, aux_ts))
        else:
            n_points = len(args["shift_index"])
            shift_index_flipped = args["shift_index"][::-1]
            shift_index_flipped = list(shift_index_flipped)
            _, sl, _ = x.shape
            step = 1 / (n_points + 1)
            for i, index in enumerate(shift_index_flipped):
                if index == '1':
                    i += 1
                    aux_ts = time_shift_surrogates(x, int(sl * i * step))
                    aux_x = np.concatenate((aux_x, aux_ts))
    if args["TI"]:
        aux_ti = time_inverse_surrogates(x)
        aux_x = np.concatenate((aux_x, aux_ti))
    if args["AI"]:
        aux_ai = amplitude_inverse_surrogates(x)
        aux_x = np.concatenate((aux_x, aux_ai))
    if args["CO"]:
        aux_co = crossover_surrogates(x, 100)
        aux_x = np.concatenate((aux_x, aux_co))

    # compositions
    if args["TS_TI"]:
        aux1 = time_shift_surrogates(x, 0, random_shift=True)
        aux2 = time_inverse_surrogates(aux1)
        aux_x = np.concatenate((aux_x, aux2))
    if args["TI_TS"]:
        aux1 = time_inverse_surrogates(x)
        aux2 = time_shift_surrogates(aux1, 0, random_shift=True)        
        aux_x = np.concatenate((aux_x, aux2))
    if args["TS_AI"]:
        aux1 = time_shift_surrogates(x, 0, random_shift=True)
        aux2 = amplitude_inverse_surrogates(aux1)
        aux_x = np.concatenate((aux_x, aux2))
    if args["TI_AI"]:
        aux1 = time_inverse_surrogates(x)
        aux2 = amplitude_inverse_surrogates(aux1)
        aux_x = np.concatenate((aux_x, aux2))
    if args["TS_TI_AI"]:
        aux1 = time_shift_surrogates(x, 0, random_shift=True)
        aux2 = time_inverse_surrogates(aux1)
        aux3 = amplitude_inverse_surrogates(aux2)
        aux_x = np.concatenate((aux_x, aux3))
    if args["TI_TS_AI"]:
        aux1 = time_inverse_surrogates(x)
        aux2 = time_shift_surrogates(aux1, 0, random_shift=True)
        aux3 = amplitude_inverse_surrogates(aux2)        
        aux_x = np.concatenate((aux_x, aux3))
        
    aux_y = np.full(len(aux_x), -99)

    x = np.concatenate((x, aux_x))
    y = np.concatenate((y, aux_y))
    return x, y


def read_data(dataset, familiy=None, fold=False):
    if dataset == "ztf":
        if familiy == "periodic":
            template = "../datasets/ztf/periodic/{}"
        elif familiy == "stochastic":
            template = "../datasets/ztf/stochastic/{}"
        elif familiy == "transient":
            template = "../datasets/ztf/transient/{}"
    elif dataset == "asas_sn":
        template = "../datasets/asas_sn/{}"
    if fold:        
        x_train = np.load(template.format("pf_train.npy"))
        x_test = np.load(template.format("pf_test.npy"))
        x_val = np.load(template.format("pf_val.npy"))
        p_train = np.load(template.format("p_train.npy"))
        p_test = np.load(template.format("p_test.npy"))
        p_val = np.load(template.format("p_val.npy"))
    else:
        x_train = np.load(template.format("x_train.npy"))
        x_test = np.load(template.format("x_test.npy"))
        x_val = np.load(template.format("x_val.npy"))
    y_train = np.load(template.format("y_train.npy"))
    y_test = np.load(template.format("y_test.npy"))
    y_val = np.load(template.format("y_val.npy"))
    if fold:
        return x_train, x_test, x_val, y_train, y_test, y_val, p_train, p_test, p_val
    else:
        return x_train, x_test, x_val, y_train, y_test, y_val


def time_norm(x, log=False, dt=False):
    mask = (x[:, :, 2] != 0)
    tmin = x[:, 0, 0][:, np.newaxis]
    x[:, :, 0] = (x[:, :, 0] - tmin) * mask
    if dt:
        x[:, 1:, 0] = (x[:, 1:, 0] - x[:, :-1, 0]) * mask[:, 1:]
    if log:
        x[:, :, 0] = np.log10(x[:, :, 0] + 1e-10) * mask    
    x[:, 0, 0] = 0
    return x


def filter_inlier_data(x_train, y_train, outlier_class):
    # filter by inlier class for training
    inlier_filter = y_train != outlier_class
    x_train = x_train[inlier_filter]
    y_train = y_train[inlier_filter]
    return x_train, y_train


def change_label(y_train, y_test, outlier_class):
    new2old = np.unique(y_train)
    old2new = {lab: i for i, lab in enumerate(new2old)}
    old2new[outlier_class] = - 99
    y_train = np.array([old2new[lab] for lab in y_train])
    mask = y_test == outlier_class
    y_test = np.array([old2new[lab] for lab in y_test])
    return y_train, y_test


def normalize_light_curves(x, eps=1e-10, minmax=False, pob=False, mean=None, std=None):
    seq_len = (x[:, :, 2] != 0).sum(axis=-1)
    if pob:
        for i in range(len(x)):
            xi = x[i, :seq_len[i], 1]
            x[i, :seq_len[i], 1] = (x[i, :seq_len[i], 1] - mean) / (std + eps)
            if x.shape[2] == 3:
                x[i, :seq_len[i], 2] = x[i, :seq_len[i], 2] / (std + eps)
        means = 0
        stds = 0
    else:        
        means = np.zeros(len(x)) 
        stds = np.zeros(len(x))
        for i in range(len(x)):
            xi = x[i, :seq_len[i], 1]
            mean = xi.mean()
            std = xi.std()
            if minmax:
                y1 = 1
                y0 = -1
                xmin = xi.min()
                xmax = xi.max()
                delta_y = y1 - y0
                delta_x = xmax - xmin
                x[i, :seq_len[i], 1] = delta_y / delta_x * (x[i, :seq_len[i], 1] - xmin) + y0
                if x.shape[2] == 3:
                    x[i, :seq_len[i], 2] = delta_y / delta_x * x[i, :seq_len[i], 2]
            else:
                x[i, :seq_len[i], 1] = (x[i, :seq_len[i], 1] - mean) / (std + eps)
                if x.shape[2] == 3:
                    x[i, :seq_len[i], 2] = x[i, :seq_len[i], 2] / (std + eps)
            means[i] = mean
            stds[i] = std
        means = means[:, np.newaxis]
        stds = stds[:, np.newaxis]
    return x, means, stds


def process_geotrans(x, args):
    index = np.arange(len(x))
    np.random.shuffle(index)
    x = x[index]

    aux_x = np.empty((0, x.shape[1], x.shape[2]))
    aux_y = np.empty((0))

    if args.AI:
        aux_ai = amplitude_inverse_surrogates(x)
        aux_x = np.concatenate((aux_x, aux_ai))
        aux_y = np.concatenate((aux_y, np.full(len(aux_ai), 1)))
    if args.TI:
        aux_ti = time_inverse_surrogates(x)
        aux_x = np.concatenate((aux_x, aux_ti))
        aux_y = np.concatenate((aux_y, np.full(len(aux_ti), 2)))
    if args.TS:
        aux_ts = time_shift_surrogates(x, 100)
        aux_x = np.concatenate((aux_x, aux_ts))
        aux_y = np.concatenate((aux_y, np.full(len(aux_ts), 3)))
    if args.CO:
        aux_co = crossover_surrogates(x, 100)
        aux_x = np.concatenate((aux_x, aux_co))
        aux_y = np.concatenate((aux_y, np.full(len(aux_co), 4)))

    # # TI-TS
    # aux_tits = time_inverse_surrogates(x)
    # aux_tits = time_shift_surrogates(aux_tits, 100)
    # aux_x = np.concatenate((aux_x, aux_tits))
    # aux_y = np.concatenate((aux_y, np.full(len(aux_tits), 4)))

    # # TS-TI
    # aux_tsti = time_inverse_surrogates(x)
    # aux_tsti = time_shift_surrogates(aux_tsti, 100)
    # aux_x = np.concatenate((aux_x, aux_tsti))
    # aux_y = np.concatenate((aux_y, np.full(len(aux_tsti), 5)))

    # # TI-AI
    # aux_tiai = time_inverse_surrogates(x)
    # aux_tiai = amplitude_inverse_surrogates(aux_tiai, 100)
    # aux_x = np.concatenate((aux_x, aux_tiai))
    # aux_y = np.concatenate((aux_y, np.full(len(aux_tiai), 6)))

    # # TS-AI
    # aux_tsai = time_shift_surrogates(x, 100)
    # aux_tsai = amplitude_inverse_surrogates(aux_tsai)
    # aux_x = np.concatenate((aux_x, aux_tsai))
    # aux_y = np.concatenate((aux_y, np.full(len(aux_tsai), 7)))

    # # TI-TS-AI
    # aux_titsai = time_inverse_surrogates(x)
    # aux_titsai = time_shift_surrogates(aux_titsai, 100)
    # aux_titsai = amplitude_inverse_surrogates(aux_titsai)
    # aux_x = np.concatenate((aux_x, aux_titsai))
    # aux_y = np.concatenate((aux_y, np.full(len(aux_titsai), 8)))

    # # TS-TI-AI
    # aux_tstiai = time_inverse_surrogates(x)
    # aux_tstiai = time_shift_surrogates(aux_tstiai, 100)
    # aux_tstiai = amplitude_inverse_surrogates(aux_tstiai)
    # aux_x = np.concatenate((aux_x, aux_tstiai))
    # aux_y = np.concatenate((aux_y, np.full(len(aux_tstiai), 9)))

    y = np.concatenate((np.full(len(x), 0), aux_y))
    x = np.concatenate((x, aux_x))
    return x, y


def remove_data_from_selected_class(x, y, l):
    mask = y != l
    return x[mask], y[mask]