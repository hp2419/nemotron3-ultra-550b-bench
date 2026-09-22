# Raw guidellm output

Drop the per-run JSON here. Filenames follow `<config>_<isl>_<osl>.json`:

```
baseline_512_256.json
mtp2_512_256.json
seqs320_512_256.json
seqs320_mtp2_512_256.json
baseline_14000_5000.json
mtp2_14000_5000.json
seqs320_mtp2_14000_5000.json
```

Copy them out of the benchmark pod with:

```bash
oc cp <namespace>/<pod>:/workspace/results ./results/raw
```
