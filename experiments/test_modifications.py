"""
Simple test script to verify the modifications work correctly.
Run this to ensure the refactored code functions as expected.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gully_performance_exp import run_label_aggregation_methods, print_results_table


def test_basic_functionality():
    """Test basic functionality with synthetic data."""
    print("="*80)
    print("TEST 1: Basic Functionality")
    print("="*80)
    
    # Create simple synthetic data
    np.random.seed(42)
    n_samples = 100
    n_lfs = 5
    
    # Generate label matrix with some abstentions
    X = np.random.choice([-1, 0, 1], size=(n_samples, n_lfs), p=[0.2, 0.4, 0.4])
    
    # Generate ground truth
    y = np.random.choice([0, 1], size=n_samples)
    
    print(f"Data shape: {X.shape}")
    print(f"Ground truth shape: {y.shape}")
    print(f"Abstention rate: {np.mean(X == -1):.2%}")
    
    try:
        # Run methods (skip LELA for faster testing)
        results, times = run_label_aggregation_methods(
            X=X,
            y=y,
            dataset_name="test_basic",
            use_f1=False,
            lela_checkpoint=None
        )
        
        print("\n✓ Test passed: Basic functionality works")
        print(f"  Methods run: {list(results.keys())}")
        print(f"  All methods returned scores: {all(0 <= v <= 1 for v in results.values())}")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_f1_score():
    """Test F1 score option."""
    print("\n" + "="*80)
    print("TEST 2: F1 Score Option")
    print("="*80)
    
    np.random.seed(43)
    X = np.random.choice([-1, 0, 1], size=(100, 5), p=[0.2, 0.4, 0.4])
    y = np.random.choice([0, 1], size=100)
    
    try:
        results, times = run_label_aggregation_methods(
            X=X,
            y=y,
            dataset_name="test_f1",
            use_f1=True,  # Test F1 score
            lela_checkpoint=None
        )
        
        print("\n✓ Test passed: F1 score option works")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False


def test_results_table():
    """Test results table printing."""
    print("\n" + "="*80)
    print("TEST 3: Results Table Printing")
    print("="*80)
    
    np.random.seed(44)
    
    # Create multiple datasets
    all_results = {}
    all_times = {}
    
    for i, dataset_name in enumerate(["dataset_A", "dataset_B"]):
        X = np.random.choice([-1, 0, 1], size=(50, 5), p=[0.2, 0.4, 0.4])
        y = np.random.choice([0, 1], size=50)
        
        results, times = run_label_aggregation_methods(
            X=X,
            y=y,
            dataset_name=dataset_name,
            use_f1=False,
            lela_checkpoint=None
        )
        
        all_results[dataset_name] = results
        all_times[dataset_name] = times
    
    try:
        # Test table printing
        df = print_results_table(all_results, all_times)
        
        print("\n✓ Test passed: Results table printing works")
        print(f"  Table shape: {df.shape}")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*80)
    print("TEST 4: Edge Cases")
    print("="*80)
    
    # Test 1: All abstentions in one LF
    print("\nTest 4a: LF with all abstentions")
    np.random.seed(45)
    X = np.random.choice([-1, 0, 1], size=(50, 5), p=[0.2, 0.4, 0.4])
    X[:, 0] = -1  # First LF abstains on everything
    y = np.random.choice([0, 1], size=50)
    
    try:
        results, times = run_label_aggregation_methods(
            X=X, y=y, dataset_name="test_all_abstain", lela_checkpoint=None
        )
        print("✓ Handles LF with all abstentions")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 2: Some samples with all abstentions
    print("\nTest 4b: Samples with all abstentions")
    X = np.random.choice([-1, 0, 1], size=(50, 5), p=[0.2, 0.4, 0.4])
    X[0, :] = -1  # First sample has all abstentions
    y = np.random.choice([0, 1], size=50)
    
    try:
        results, times = run_label_aggregation_methods(
            X=X, y=y, dataset_name="test_sample_abstain", lela_checkpoint=None
        )
        print("✓ Handles samples with all abstentions")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print("\n✓ Test passed: Edge cases handled correctly")
    return True


def test_multiclass():
    """Test multiclass classification."""
    print("\n" + "="*80)
    print("TEST 5: Multiclass Classification")
    print("="*80)
    
    np.random.seed(46)
    
    # 3-class problem
    X = np.random.choice([-1, 0, 1, 2], size=(100, 5), p=[0.2, 0.27, 0.27, 0.26])
    y = np.random.choice([0, 1, 2], size=100)
    
    print(f"Number of classes: {len(np.unique(y))}")
    
    try:
        results, times = run_label_aggregation_methods(
            X=X,
            y=y,
            dataset_name="test_multiclass",
            use_f1=False,
            lela_checkpoint=None
        )
        
        print("\n✓ Test passed: Multiclass classification works")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS")
    print("="*80 + "\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("F1 Score Option", test_f1_score),
        ("Results Table", test_results_table),
        ("Edge Cases", test_edge_cases),
        ("Multiclass", test_multiclass),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\nUnexpected error in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The modifications are working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
