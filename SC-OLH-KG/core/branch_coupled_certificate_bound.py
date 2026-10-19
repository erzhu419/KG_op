"""A marginal branch bound retaining primary/incumbent event budgets."""
import numpy as np

from core.marginal_certificate_bound import branch_quality_upper


def coupled_branch_quality_upper(quality, marginals, incumbent):
    """Solve the three-policy/four-branch LP through feasible duals.

    The primal adds z_primary+z_pair<=u_primary and
    z_incumbent+z_pair<=u_incumbent to the old mutually exclusive branches.
    For total-mass dual t, the other duals are positive quality deficits.
    Any paired deficit is assigned to the cheaper of the two marginal caps.
    Breakpoints are0,qP,qI,qB,qPI,qP+qI-qPI; no per-action LP solver is needed.
    """
    quality=np.asarray(quality,dtype=float)
    marginals=np.broadcast_to(np.asarray(marginals,dtype=float),quality.shape[:-1]+(3,))
    assert quality.shape[-1]==4 and incumbent in (0,1,2)
    if incumbent==0:
        return branch_quality_upper(quality,marginals,incumbent)
    fallback=3-incumbent
    qP,qI,qB,qPI=[quality[...,j,None] for j in (0,incumbent,fallback,3)]
    uP,uI,uB=[marginals[...,j,None] for j in (0,incumbent,fallback)]
    t=np.concatenate([np.zeros_like(qP),qP,qI,qB,qPI,qP+qI-qPI],axis=-1)
    t=np.maximum(t,0.)
    yP=np.maximum(qP-t,0.);yI=np.maximum(qI-t,0.);yB=np.maximum(qB-t,0.)
    extra=np.maximum(qPI-t-yP-yI,0.)
    cheaper=uP<=uI
    yP+=np.where(cheaper,extra,0.);yI+=np.where(cheaper,0.,extra)
    # Raising the total-mass dual fixes any arithmetic residual and preserves
    # an upper bound; padding also covers the small positive objective sum.
    residual=np.maximum.reduce([qP-yP-t,qI-yI-t,qB-yB-t,qPI-yP-yI-t,np.zeros_like(t)])
    padding=16*np.finfo(float).eps*np.maximum(1.,np.max(np.abs(quality),axis=-1,keepdims=True))
    t=t+residual+padding
    return np.min(t+uP*yP+uI*yI+uB*yB,axis=-1)
