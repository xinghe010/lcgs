from .augmentation import transitivity_augmentation
from .normalization import normalize_ast, normalize_code_semantics
from .compatibility import (is_compatible, get_canonical_label,
                            TYPE_HIERARCHY, get_type_hierarchy)
from .tptf import (compute_tptf_vector, compute_corpus_idf,
                   collect_token_depths, get_ast_helpers, get_type_tiers)
from .cwj import (compute_cwj, check_mcs_equal, get_wl_features,
                  ast_to_networkx, compute_pair_features)

try:
    from .mgda import MinNormSolver, MGDA
    from .lcgs_trainer import LCGSTrainer
except ImportError:
    pass
