import argparse
import gzip
import json
import os



parser = argparse.ArgumentParser()
parser.add_argument('--auth_file', required=True, help='Path to the auth.txt.gz file')
parser.add_argument('--proc_file', required=True, help='Path to the proc.txt.gz file')
parser.add_argument('--redteam_file', required=True, help='Path to the proc.txt.gz file')
parser.add_argument('--output_dir', default='lanl_dataset/', help='Path where the output files should be written')
parser.add_argument('--proc_count_cutoff', type=int, default=40, help='Minimum number of occurrences for a process name to be included')
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


# Fetch red team events
rt = []
with gzip.open(args.redteam_file, 'rt') as f:
    for l in f:
        rt.append(l.strip())

u_dict = {}
h_dict = {}
p_dict = {}
t_dict = {}

train_file = os.path.join(args.output_dir, 'train.csv')
test_file = os.path.join(args.output_dir, 'test.csv')

# Fetch and preprocess authentication events
with gzip.open(args.auth_file, 'rt') as f:
    print('Current file: {0}'.format(args.auth_file))
    lines = []
    for l in f:
        tmp = l.strip().split(',')
        if int(tmp[0]) >= 1123200:
            break
        if tmp[7] == 'LogOn' and tmp[2].startswith('U'):
            usr, src, dst = tmp[2:5]
            if tmp[5].startswith('MICROSOFT_'):
                tmp[5] = 'MSV_1_0'
            lt = '_'.join(tmp[5:7])
            if usr not in u_dict:
                u_dict[usr] = len(u_dict)
            for x in (src, dst):
                if x not in h_dict:
                    h_dict[x] = len(h_dict)
            if lt not in t_dict:
                t_dict[lt] = len(t_dict)
            x = ','.join([tmp[0], usr, src, dst])
            if x in rt:
                lab = 1
            else:
                lab = 0
            if src == dst:
                line = ','.join([
                    str(u_dict[usr]), str(t_dict[lt]), str(h_dict[src]),
                    '0', tmp[0], '0', str(lab)
                ])
            else:
                line = ','.join([
                    str(u_dict[usr]), str(t_dict[lt]), str(h_dict[src]),
                    str(h_dict[dst]), tmp[0], '1', str(lab)
                ])
            lines.append(line)
            if len(lines) >= 100000:
                lines = list(set(lines))
                with open(train_file, 'a+') as out_train, \
                    open(test_file, 'a+') as out_test:
                    for line in lines:
                        if int(line.split(',')[4]) < 691200:
                            out_train.write(line + '\n')
                        else:
                            out_test.write(line + '\n')
                lines = []

lines = list(set(lines))
with open(train_file, 'a+') as out_train, \
    open(test_file, 'a+') as out_test:
    for line in lines:
        if int(line.split(',')[4]) < 691200:
            out_train.write(line + '\n')
        else:
            out_test.write(line + '\n')

# Look for rare process names
with gzip.open(args.proc_file, 'rt') as f:
    print('Current file: proc.txt.gz')
    print('Counting occurrences...')
    proc = {}
    for l in f:
        tmp = l.strip().split(',')
        if int(tmp[0]) >= 1123200:
            break
        if tmp[4] == 'Start' and tmp[1].startswith('U'):
            p = tmp[3]
            if p not in proc:
                proc[p] = 0
            proc[p] += 1

p_counts = []
rare_proc = set()
rare_count = 0
for p in proc:
    if proc[p] < args.proc_count_cutoff:
        rare_proc.add(p)
        rare_count += proc[p]
    else:
        p_counts.append((p, proc[p]))
p_counts.append(('RP', rare_count))
p_counts.sort(key=lambda x: x[1], reverse=True)
p_dict = dict(zip([p[0] for p in p_counts], list(range(len(p_counts)))))

# Fetch and preprocess process creation events
with gzip.open(args.proc_file, 'rt') as f:
    print('Current file: proc.txt.gz')
    lines = []
    for l in f:
        tmp = l.strip().split(',')
        if int(tmp[0]) >= 1123200:
            break
        if int(tmp[0]) >= 691200:
            current_output = test_file
        else:
            current_output = train_file
        if tmp[4] == 'Start' and tmp[1].startswith('U'):
            usr, hst, prc = tmp[1:4]
            if prc in rare_proc:
                prc = 'RP'
            if usr not in u_dict:
                u_dict[usr] = len(u_dict)
            if hst not in h_dict:
                h_dict[hst] = len(h_dict)
            if prc not in p_dict:
                p_dict[prc] = len(p_dict)
            lab = 0
            line = ','.join([
                str(h_dict[hst]), str(u_dict[usr]), str(p_dict[prc]),
                '0', tmp[0], '2', '0'
            ])
            lines.append(line)
            if len(lines) >= 100000:
                lines = list(set(lines))
                with open(train_file, 'a+') as out_train, \
                    open(test_file, 'a+') as out_test:
                    for line in lines:
                        if int(line.split(',')[4]) < 691200:
                            out_train.write(line + '\n')
                        else:
                            out_test.write(line + '\n')
                lines = []

lines = list(set(lines))
with open(train_file, 'a+') as out_train, \
    open(test_file, 'a+') as out_test:
    for line in lines:
        if int(line.split(',')[4]) < 691200:
            out_train.write(line + '\n')
        else:
            out_test.write(line + '\n')

# Write entity-index maps
names = ['lanl_hosts', 'lanl_users', 'lanl_proc', 'lanl_types']
data = [h_dict, u_dict, p_dict, t_dict]
for n, d in zip(names, data):
    fp = os.path.join(args.output_dir, n + '.json')
    with open(fp, 'w') as out:
        out.write(json.dumps(d))
