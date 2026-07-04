import Lake
open Lake DSL

package «scolhkg-proof» where
  version := v!"0.1.0"

@[default_target]
lean_lib SCOLHKG where
  roots := #[`SCOLHKG]
