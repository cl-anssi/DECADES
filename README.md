

# Content


This repository contains our Python implementation of the DECADES model presented in our paper "Dynamically Modelling Heterogeneous Higher-Order Interactions for Malicious Behavior Detection in Event Logs".

The code can be found in the decades directory:

 * The utils subdirectory contains general purpose funtions that are used in other modules and classes.
 * The dataset subdirectory contains a module that handles input log files.
 * The model subdirectory contains several modules that implement the DECADES model.
 * test_decades.py is a wrapper script that launches DECADES using provided parameters and input data, and data_preprocessing.py creates the appropriately processed input files from the raw LANL dataset.

In order to reproduce the experiments presented in the paper, you must first download the LANL dataset (auth.txt.gz, proc.txt.gz and redteam.txt.gz, available at https://csr.lanl.gov/data/cyber1/).
The data_preprocessing.py and test_decades.py scripts can then be used to transform these files into appropriate input files and run the algorithm on these preprocessed inputs, respectively.






# Dependencies

The code is written in Python 3 and relies on the following libraries:

* torch >= 1.7.0
* numpy >= 1.20
* pandas >= 1.2.0



# Usage example

The following commands reproduce the experiments presented in the paper. Note that the auth.txt.gz, proc.txt.gz and redteam.txt.gz should be placed in the current working directory, and that significant computing resources (including GPU) are required to keep the running time acceptable.

```
$ python decades/data_preprocessing.py --auth_file auth.txt.gz --proc_file proc.txt.gz --redteam_file redteam.txt.gz --output_dir lanl_data/
$ sort -n -k 5 -t ',' -o lanl_data/train_sorted.csv lanl_data/train.csv
$ mv lanl_data/train_sorted.csv lanl_data/train.csv
$ sort -n -k 5 -t ',' -o lanl_data/test_sorted.csv lanl_data/test.csv
$ mv lanl_data/test_sorted.csv lanl_data/test.csv
$ gzip lanl_data/train.csv lanl_data/test.csv
$ for i in {0..19}; do
$    python decades/test_decades.py --input_dir lanl_data/ --output_dir experiments/ --conf_file decades/conf.json --return_pval --fix_seed $i
$ done
```
