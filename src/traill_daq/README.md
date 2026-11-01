# Serial acquisition

Install optional dependencies with `python -m pip install -e '.[acquisition]'`.
Run `python -m traill_daq.run --help`, then specify the actual serial port and
recording output folder:

```bash
python -m traill_daq.run --port COM5 --path /private/soma/recordings
```

The default COM31 is a historical local setting; specify your own port.
The adjacent activity/paper-tape JSON profiles are packaged resources resolved
relative to this package. `traill_visualizer.py` owns the live plotting and
serial readout. See [hardware](../../hardware/README.md) for firmware locations.
Physical serial acquisition was not tested during reorganization.
