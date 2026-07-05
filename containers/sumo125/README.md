# SUMO 1.25 Local Container

This container pins the local SUMO stack to the same major environment used for
the node001-node006 scheduler runs:

- `eclipse-sumo==1.25.0`
- `libsumo==1.25.0`
- `sumolib==1.25.0`
- `traci==1.25.0`
- Python 3.10

Build:

```bash
containers/sumo125/build.sh
```

Smoke test:

```bash
containers/sumo125/check.sh
```

Run the ingolstadt21 validator in the container:

```bash
containers/sumo125/run_ingolstadt21.sh --dry-run
containers/sumo125/run_ingolstadt21.sh --R 2 --max-points 1 --jobs 1 --backend libsumo
```

The scripts mount the repository at `/workspace` and write outputs back to the
same local checkout.
