import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sleeptcn.development import softened_class_weights, require_development_roles


def test_grid_endpoints_match_existing_controls():
    counts = np.array([54633,17152,55438,9684,20293])
    np.testing.assert_array_equal(softened_class_weights(counts,0),np.ones(5))
    np.testing.assert_allclose(softened_class_weights(counts,1),counts.sum()/(5*counts))
    ratios = [softened_class_weights(counts,p)[3]/softened_class_weights(counts,p)[2] for p in [0,.25,.5,1]]
    assert ratios == sorted(ratios)


@pytest.mark.parametrize('counts,power', [([1,2,3,4,0],.5),([1,2,3,4,5],1.1),([1,2,3,4,5],float('nan')),([1,2,3,4,5.5],.5)])
def test_invalid_weights_fail(counts,power):
    with pytest.raises(ValueError):
        softened_class_weights(counts,power)


def test_development_roles_forbid_test():
    require_development_roles([{'path':'data/train/000.npz','role':'train'},
                               {'path':'data/validation/000.npz','role':'validation'}])
    with pytest.raises(ValueError):
        require_development_roles([{'path':'data/test/000.npz','role':'test'}])
