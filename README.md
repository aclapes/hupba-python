# Hupba-python

IMPORTANT: Python3.6 or above is required.

To use all the tools install the python requirements in the root directory.

```
git clone https://github.com/aclapes/hupba-python.git
cd hupba-python
pip3 install -r requirements.txt
```

In case of only wanting to use the ones in some particular directory install those, e.g. for evaluation: 

```
git clone https://github.com/aclapes/hupba-python.git
cd hupba-python/evaluation
pip3 install -r evaluation/requirements.txt
```

## List of available tools

`calibration/`

	depth.py : calibration of other modalities to depth when rotation-translation between the cameras is known.

`evaluation/`

	metrics_1d.py : overlap and f1-score on 1-d time series.

`synchronization/`

	temporal.py : temporally match two series of timestamps.
