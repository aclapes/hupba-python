# Hupba-python

IMPORTANT: Python3.6 or above is required.

To use all the tools install the python requirements in the root directory.

```
sudo apt-get install python
git clone https://github.com/aclapes/hupba-python.git && cd hupba-python
pip3 install -r requirements.txt
```

In case of only wanting to use the ones in some particular directory install those, e.g. for evaluation: 

```
cd evaluation
pip3 install -r evaluation/requirements.txt
```

## List of available tools

`evaluation/`

	metrics_1d.py : overlap and f1-score on 1-d time series.

`synchronization/`

	temporal.py : temporally match two series of timestamps.
