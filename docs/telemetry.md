# Telemetry schema

## Model

A **signal** is identified by a dotted name and described by a `SignalSpec`:
canonical unit, kind, physical range, a plausible maximum rate of change, and how
long its last value may be held when the sensor misses a tick.

A **reading** is a value plus a `Quality`. This is the part that matters: the
system never silently fabricates a measurement, so downstream code can always
distinguish "CPU was at 0%" from "we could not read the CPU".

| Quality | Meaning |
|---|---|
| `OK` | Measured this tick |
| `CLAMPED` | Outside the declared range; clipped and logged |
| `SUSPECT` | Changed faster than physically plausible; kept, flagged |
| `STALE` | Sensor missed a tick; last value held, within `max_hold_s` |
| `IMPUTED` | Filled by the normaliser |
| `MISSING` | Genuinely unavailable |

`Quality.usable` is true up to `SUSPECT`. A forecaster only consumes usable
readings; everything else becomes NaN in the feature window and is handled
explicitly.

Storage is long-format — `readings(sample_id, signal_id, value, quality)` — so
adding a signal is an insert, never a migration.

## Signals collected on the reference machine

38 signals from 8 sources. Availability is discovered at probe time; anything
absent is dropped from the feature space rather than imputed forever.

### CPU
| signal | unit | notes |
|---|---|---|
| `cpu.util_pct` | percent | Aggregate utilisation since the previous tick |
| `cpu.util_max_core_pct` | percent | Busiest logical core — separates one pinned thread from a parallel build |
| `cpu.iowait_pct` | percent | Time blocked on I/O |
| `cpu.freq_mhz` | megahertz | Average core frequency |
| `cpu.load1_per_core` | ratio | 1-minute load ÷ logical cores, so it is comparable across machines |

### Memory
`mem.used_pct`, `mem.available_gb`, `mem.cached_gb` (page cache + reclaimable
slab — drops sharply under pressure), `mem.swap_used_pct`.

### Battery
| signal | unit | notes |
|---|---|---|
| `battery.percent` | percent | **Fractional** where the platform exposes it |
| `battery.plugged` | bool | AC present |
| `battery.charging` | bool | Actually taking charge (plugged-and-full reads 0) |
| `battery.power_w` | watt | Pack power, from `power_now` or `current_now × voltage_now` |

Read from `/sys/class/power_supply` in preference to psutil, because
`charge_now / charge_full` gives fractional state of charge. The integer
`capacity` file — which psutil reports — quantises to 1%, and at a realistic
0.15 %/min drain that turns the signal into a staircase with seven-minute treads
whose instantaneous slope is either zero or a spike. psutil remains the
cross-platform fallback.

### Thermal
`thermal.cpu_c`, `thermal.gpu_c`, `thermal.disk_c`, `thermal.wireless_c`,
`thermal.max_c`.

The kernel names sensors by driver — `k10temp` on AMD, `coretemp` on Intel,
`amdgpu`, `nvme`, `iwlwifi`. Emitting those as signal names would make every
stored dataset and trained model machine-specific, so drivers are mapped onto
portable **roles**. An unmapped driver still contributes to `thermal.max_c`
rather than being dropped.

### GPU (NVIDIA)
`gpu.util_pct`, `gpu.mem_used_pct`, `gpu.mem_used_gb`, `gpu.temp_c`,
`gpu.power_w`.

NVML when `nvidia-ml-py` is importable (0.02 ms per full read); otherwise
`nvidia-smi`, which costs 45 ms and therefore declares a 10 s cadence that the
collector decimates. The signal names are vendor-neutral, so an AMD or Apple
source can be added as a sibling emitting the same names.

### Disk and network
`disk.read_mbps`, `disk.write_mbps`, `disk.busy_pct`, `net.recv_mbps`,
`net.sent_mbps`.

The OS exposes cumulative counters, not rates. Conversion is stateful and has two
real failure modes — counter resets and non-positive time deltas — both of which
yield `None` for the affected interval rather than a fabricated spike. The first
tick after startup legitimately has no rate.

### Process attribution
`proc.count`, `proc.top1_cpu_pct`, `proc.concentration` (share of CPU held by the
top three — 1.0 means one dominant task), and `proc.cpu_<category>_pct` for
`build`, `browser`, `media`, `ml`, `container`, `editor`, `system`.

This is what lets the system tell "70% CPU because of a build" from "70% because
of a video call", which is what makes regime-aware forecasting and personalised
explanations possible.

**Privacy.** Process names reveal which applications and projects a person uses,
so this source aggregates into coarse categories and **never persists process
names or command lines**. `include_names` exists for local debugging and is off
by default. Disable the source entirely with:

```bash
streamlora collect --set collect.process_attribution=false
```

A full pass over ~500 processes costs ~40 ms, under 1% duty cycle at the default
cadence.

## Cost

On the reference machine, one tick reads all 38 signals in **~45 ms**, dominated
by process enumeration. At 5 s that is under 1% of one core.

## Adding a signal

Write one class. Nothing above it changes.

```python
class FanSource(TelemetrySource):
    name = "fan"

    def signals(self):
        return [SignalSpec("fan.rpm", "rpm", SignalKind.GAUGE, 0.0, 12000.0,
                           max_rate_per_s=3000.0)]

    def probe(self):
        # Cheap capability check. Must not raise.
        return ProbeResult(available, detail="hwmon fan1", provides=("fan.rpm",))

    def read(self, now):
        # None means "unavailable right now"; raise only if the whole source failed.
        return {"fan.rpm": value_or_none}
```

Register it in `telemetry/registry.py::default_sources`. The storage schema, the
feature extractor, the API and the dashboard all pick it up automatically. Add it
to `FeatureConfig.inputs` to make it a model input, and to `ForecastConfig.targets`
to forecast it.

If the source declares `min_interval_s`, the collector decimates it and holds the
previous value in between — intentional cadence, not staleness.

## Datasets

`streamlora export` writes newline-delimited JSON: a header line with the signal
specs, then one object per sample.

```json
{"ts": 1787086059.44, "origin": "live", "gap_s": 5.04, "collect_ms": 72.8,
 "r": {"cpu.util_pct": [5.2], "mem.used_pct": [53.1], "gpu.temp_c": [null, 5]}}
```

A reading is `[value]` when quality is `OK`, otherwise `[value, quality]`, so
degraded data survives a round trip. NDJSON was chosen over a binary format
because a dataset is the primary artefact an experiment is judged on, and being
able to `head` it, diff it and read it in five years without this codebase is
worth more than the space saving. It gzips ~10× if that matters.
