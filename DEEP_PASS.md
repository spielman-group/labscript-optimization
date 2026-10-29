# labscript-optimization deep pass

The code-health campaign's pass over this repo (`issues/code_health_campaign.md`
in the meta repo) is complete on `CodeHealth`, and nothing in it is open. This
file goes when `CodeHealth` merges into `Development`.

## Merges with

`CodeHealth` needs these, which are on the other repos' `CodeHealth` and not on
their `Development`:

- lyse `a72da38`: `lyse.Routine.folder`.
- runmanager `090fba0`: `submit_shots` takes `sequence_id`.
- runmanager `eb28e07`: `runmanager.client.bound_for_lyse`.

## Kept by decision

Each of these could look like a candidate for removal, and was kept:

- `RunmanagerInterface.check_unchanged`: runmanager refuses a sequence of
  another labscript file, but compares base names only, so a file of the same
  name in another folder passes there and not here.
- `NO_VALUE_YET`'s typed stand-ins: lyse fixes a column's dtype from the first
  value saved.
- The column sort in `OptimizationRoutine.run`: sorting in `lyse.data` would
  reorder every lab's columns.
- `GlobalMapping.evaluate`'s conversion of numpy scalars: the optimizer
  converts, not runmanager.
- `Worker` stays in this package; it becomes a `lyse.Routine` facility only once
  a second routine needs one.
- This package and labscript-agents each keep their own record of which shots
  are theirs, each taken once.
- `in_background` in the Gaussian process: no suite helper returns a `Future`
  from a daemon thread.
