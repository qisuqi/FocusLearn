import pandas as pd


def compute_usage_intervals(ts, d_max):
    ts = pd.to_datetime(ts, errors='coerce')
    ts_len = len(ts)
    d_max = int(d_max)
    current_interval = 0
    intervals = []
    intervals.append([])
    durations = []

    for i in range(0, ts_len - 1):

        distance = abs((ts[i + 1] - ts[i]).total_seconds())

        if distance <= d_max:
            intervals[current_interval].append(ts[i + 1])
        else:
            current_interval += 1
            intervals.append([])
            intervals[current_interval].append(ts[i + 1])

    intervals[0].insert(0, ts[0])

    for date in intervals:
        dr = (date[-1] - date[0]).total_seconds()
        durations.append((date[0].strftime('%Y-%m-%d'), dr))

    return durations


def preprocess_multivariate_ts(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-d%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    aggs = pd.concat(cols, axis=1)
    aggs.columns = names

    if dropnan:
        aggs.dropna(inplace=True)
    return aggs


def remove_static_cols(col_names, timesteps=3):

    t_0 = [f'{x}_t0' for x in col_names]
    t_0[0] = 'ID'
    t_0[1] = 'Date'
    #t_0[2] = 'M'
    #t_0[3] = 'F'
    #t_0[4] = 'Age'

    t_1 = [f'{x}_t1' for x in col_names]
    t_1.remove('ID_t1')
    t_1.remove('Date_t1')
    #t_1.remove('M_t1')
    #t_1.remove('F_t1')
    #t_1.remove('Age_t1')

    t_2 = [f'{x}_t2' for x in col_names]
    t_2.remove('ID_t2')
    t_2.remove('Date_t2')
    #t_2.remove('M_t2')
    #t_2.remove('F_t2')
    #t_2.remove('Age_t2')

    t_3 = [f'{x}_t3' for x in col_names]
    t_3.remove('ID_t3')
    t_3.remove('Date_t3')
    #t_3.remove('M_t3')
    #t_3.remove('F_t3')
    #t_3.remove('Age_t3')

    if timesteps == 2:
        column_names = sum([t_0, t_1], [])
    elif timesteps == 3:
        column_names = sum([t_0, t_1, t_2], [])
    else:
        raise Exception('Too many timesteps')
    return column_names
