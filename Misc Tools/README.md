# Tools Info

## Processing CICIDS2017 dataset
1. Place them within a directory
2. Run combine-csv.py (combines each seperate dataset files into a single one)
3. Run cicids-remove-dup.py (remove duplicate column)
4. Run dataset-scale.py (optional, downsize the dataset to reduce training time)
5. Run class-remap.py (optional, combine attack classes to reduce training time)

## Creating your own dataset with CICFlowMeter
1. Capture and save packets in *.pcap format using either tcpdump or wireshark.
2. Install CICFlowMeter from [here](https://github.com/LazyWraith/py-cicflowmeter-fix) (tested in Ubuntu, requires Python 3)
3. Label your capture (benign or attack etc.) using CICLabeler.py. See usage below.

## File usage

`cicids-remove-dup.py`

Remove duplicate column "Fwd Header Length.1" in original CICIDS2017 dataset.

`CICLabeler.py`

Add labels to csv files produced by CICFlowMeter.
Usage: python CICLabeler.py -i [input_file] -o [output_file] -l [label_name]

`class-remap.py`

Remap CICIDS2017 class label as follows:
```python
class_mapping = {
    'BENIGN': 'BENIGN',
    'Bot': 'Bot',
    'DDoS': 'DoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'PortScan',
    'FTP-Patator': 'BruteForce',
    'SSH-Patator': 'BruteForce',
    'Web Attack - Brute Force': 'Web Attack',
    'Web Attack - XSS': 'Web Attack',
    'Web Attack - Sql Injection': 'Web Attack',
    'Infiltration': 'Others',
    'Heartbleed': 'Others'
}
```

`combine-csv.py`

Combines all csv file within the same directory into a single file. Place this file within a folder containing only CICIDS2017 csv files (Monday to Friday) to combine them into a single file.

`dataset-scale.py`

Scale down the dataset size. Edit the target label and scaling factor within the Python file itself.
Scale factor: 'BENIGN' = 0.05, 'DoS Hulk' = 0.1, 'PortScan' = 0.1, 'DDoS' = 0.1