# Meta-prior manuscript status

## Current artifact

The legacy atlas article remains unchanged at `manuscript/main.tex` and
`manuscript/main.pdf`.  A separate bounded-claim manuscript is now assembled
at:

- source: `manuscript/meta_prior_main.tex`
- compiled PDF: `manuscript/meta_prior_main.pdf`
- sections: `manuscript/meta_prior_sections/`
- supplement source: `manuscript/meta_prior_supplement.tex`
- supplement PDF: `manuscript/meta_prior_supplement.pdf`

The new draft compiles under the INFORMS Operations Research class to an
8-page PDF.  Its supplement contains the frozen protocol, generated taskwise
paired controls, the independent v2 block, risk/representation diagnostics,
and a claim audit.  BibTeX and cross-references resolve; the remaining log
warnings are font substitutions from the class/font combination.

## What the draft claims

The draft claims a source-learned finite structural prior in the declared
observable representation and dictionary.  It reports early objective and
constraint prediction gains, target correction of mismatched structure, the
paired source-structure versus same-source-full-dictionary development
control, the independent v2 non-replication, and the negative risk/deployment
results.

It explicitly does not claim learned trajectory embeddings, separate physical
risk-factor recovery, stable independent deployment superiority, or arbitrary
high-dimensional free-variable optimization.

## Remaining submission work

1. The artifact manifest, table generator, replay entrypoint, and final claim
   audit are now connected to the new entry point; run the replay after any
   manuscript edit.
2. The original confirmation interval was not run as a block, and its declared
   interval overlaps development at seed 95434.  The separate v2 block is the
   clean fixed-seed check and does not support a stable deployment claim.
3. The trajectory-representation and factor-level-risk claims remain open
   research gaps and must stay outside the abstract and contribution list.
4. The old atlas and new meta-prior evidence have different contracts and must
   remain separate in the final submission.

The next manuscript step is a final clean replay and package-level OR format
check, not another unregistered experiment matrix.
