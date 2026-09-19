import unittest

from src.data import (
    build_sparniche_graph,
    build_sparniche_negative_mask,
    prepare_sparniche_rna_features,
)
from src.models import SparNiche, SparNicheEncoder, SparNicheOutput


class SparNicheNamingTests(unittest.TestCase):
    def test_public_model_symbols_use_sparniche_name(self):
        self.assertEqual(SparNiche.__name__, "SparNiche")
        self.assertEqual(SparNicheEncoder.__name__, "SparNicheEncoder")
        self.assertEqual(SparNicheOutput.__name__, "SparNicheOutput")

    def test_public_graph_and_preprocessing_symbols_use_sparniche_name(self):
        self.assertEqual(build_sparniche_graph.__name__, "build_sparniche_graph")
        self.assertEqual(
            build_sparniche_negative_mask.__name__, "build_sparniche_negative_mask"
        )
        self.assertEqual(
            prepare_sparniche_rna_features.__name__, "prepare_sparniche_rna_features"
        )

if __name__ == "__main__":
    unittest.main()
