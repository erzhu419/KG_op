import Lake
open Lake DSL

package «scolhkg-proof» where
  version := v!"0.1.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0"

@[default_target]
lean_lib SCOLHKG where
  roots := #[`SCOLHKG]
