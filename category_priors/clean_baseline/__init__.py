"""Clean SAGA-asset automatic-instance baseline.

Only the independent alpha-mask evidence contract is exported here.  Importing
this package has no CUDA, legacy post-processing, HDBSCAN, or ObjectBank side
effects.
"""

from .evidence import (
    accumulate_alpha_mass_from_contributors,
    build_alpha_mask_evidence,
    build_frame_evidence,
    build_sparse_frame_evidence,
    evidence_bank_is_complete,
    evidence_request_source,
    load_evidence_bank,
    save_evidence_bank,
)
from .models import (
    AlphaMaskEvidenceBank,
    AlphaMassFrame,
    EvidenceThresholds,
    FrameEvidence,
    FrameMetadata,
    MaskMetadata,
    MaskSupportCSR,
    PackedVisibility,
)

__all__ = [
    "AlphaMaskEvidenceBank",
    "AlphaMassFrame",
    "EvidenceThresholds",
    "FrameEvidence",
    "FrameMetadata",
    "MaskMetadata",
    "MaskSupportCSR",
    "PackedVisibility",
    "accumulate_alpha_mass_from_contributors",
    "build_alpha_mask_evidence",
    "build_frame_evidence",
    "build_sparse_frame_evidence",
    "evidence_bank_is_complete",
    "evidence_request_source",
    "load_evidence_bank",
    "save_evidence_bank",
]
