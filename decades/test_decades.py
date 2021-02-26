import argparse
import gzip
import json
import os
import time

import torch

from model.decades import DECADES
from dataset.dataset import LANLDataset
from utils.training_procedures import train_model, test_model

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True, help='Path to the input files')
parser.add_argument('--output_dir', default=None, help='Path where the output should be written (if None, input directory is used)')
parser.add_argument('--conf_file', default='conf.json', help='Path to the JSON file containing the configuration')
parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of the embedding space')
parser.add_argument('--neg_samples', type=int, default=20, help='Number of negative samples to draw for each batch')
parser.add_argument('--batch_size', type=int, default=5000, help='Size of the training batches (initial training)')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularization coefficient for Adam (initial training)')
parser.add_argument('--lambda_0', type=float, default=1e-4, help='Regularization coefficient for the embeddings of new entities')
parser.add_argument('--lambda_1', type=float, default=1, help='Baseline regularization coefficient for the evolution of embeddings between two timesteps')
parser.add_argument('--retrain_batch_size', type=int, default=256, help='Size of the training batches when retraining the model')
parser.add_argument('--noise_dist', default='log', help='Noise distribution to use: unigram on all train events (all) or on unique events only (unique), or log-unigram on all train events (log)')
parser.add_argument('--no_double', dest='no_double', action='store_true', help='If true, only one occurrence of each entity tuple is included in the training set for each interaction type')
parser.add_argument('--return_pval', dest='return_pval', action='store_true', help='If true, the mean negative logarithm of the mid-p-values is used as anomaly score')
parser.add_argument('--fix_seed', type=int, default=None, help='Fixed seed for the RNG (for reproducibility)')
parser.set_defaults(no_double=False, return_pval=False)
args = parser.parse_args()

if args.output_dir is not None:
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
else:
    output_dir = args.input_dir

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if args.fix_seed is not None:
    torch.manual_seed(args.fix_seed)
    torch.cuda.manual_seed(args.fix_seed)

exp = [
    str(int(1000*time.time())), 'decades', str(args.embedding_dim),
    str(args.neg_samples), str(args.batch_size),
    str(args.epochs), str(args.weight_decay), args.noise_dist,
    str(args.lambda_0), str(args.lambda_1)]
for name, val in zip(
    ('nodouble', 'pval'),
    (args.no_double, args.return_pval)):
    if val:
        exp.append(name)
exp_name = '_'.join(exp)

conf = json.loads(open(args.conf_file).read())

fp_train = os.path.join(args.input_dir, 'train.csv.gz')
fp_test = os.path.join(args.input_dir, 'test.csv.gz')

dataset = LANLDataset(
    gzip.open(fp_train, 'rt'),
    gzip.open(fp_test, 'rt'),
    conf, args.no_double)

entities = conf['entities']
for i, e in enumerate(entities):
    e['n_entities'] = dataset.arities[i]

interactions = conf['interactions']

for i in range(len(interactions)):
    noise_dist = []
    cnt = dataset.get_unigram_counts(i, unique=args.no_double)
    for j in range(1, len(interactions[i]['entities'])):
        n_entities = entities[interactions[i]['entities'][j]]['n_entities']
        counts = torch.IntTensor([
            cnt[j-1][k] if k in cnt[j-1] else 0 for k in range(n_entities)])
        noise_dist.append(counts)
    interactions[i]['noise_dist'] = noise_dist

model = DECADES(entities, interactions,
    embedding_dim=args.embedding_dim, n_noise_samples=args.neg_samples,
    noise_dist=args.noise_dist, return_pval=args.return_pval
    ).to(device)

train_start_time = time.time()

running_losses, test_scores, logvars = train_model(
    model, dataset, device, args.epochs, args.batch_size,
    weight_decay=args.weight_decay)

train_duration = time.time() - train_start_time
itr_weights = [
    [l.weights.detach().cpu().numpy().tolist() for l in itr.linear]
    for itr in model.interactions]

res = test_model(model, dataset, device, args.retrain_batch_size,
    args.lambda_0, args.lambda_1)

res_dict = {'preds': list(res[:,0]), 'labels': list(res[:,1]), 'types': list(res[:,2]),
    'losses': running_losses, 'test': test_scores, 'time': train_duration, 'logvars': logvars,
    'weights': itr_weights}
fp = os.path.join(output_dir, 'res_{0}.json'.format(exp_name))
with open(fp, 'a+') as out:
    out.write(json.dumps(res_dict))
