# Strategic PufferLib 4 Ocean Benchmark

Backend: `pufferlib_4_ocean_standalone`
Steps/sec: `4070980.81`
Envs: `512`
Steps: `100000`
Episodes: `1537`
Mean placement: `6.763`

## Puffer 4 Build

- Not attempted.

## Limits

- standalone benchmark exercises the Ocean C env loop, not the full Puffer trainer
- the env mirrors strategic rules for throughput exploration; scalar parity is covered by Python/native tests
- PufferLib 4.0 training requires building the env inside a PufferLib 4.0 source checkout
