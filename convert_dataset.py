from util import save_1dim_arrays

time, length, anom = [], [], []
with open('time.txt', 'r') as r:
    for line in r.readlines():
        time += [float(line)]
with open('length.txt', 'r') as r:
    for line in r.readlines():
        length += [int(line)]
with open('info.txt', 'r') as r:
    for line in r.readlines():
        anom += [int('DoS' in line)]
        if 'DoS' not in line and 'ok' not in line:
            raise Exception

d = {'time':time, 'length':length, 'anom':anom}
save_1dim_arrays(d, 'converted_dataset.csv')
