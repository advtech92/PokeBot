import numpy as np

def test_numpy():
    try:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.abs(a - b)
        print("NumPy is working correctly. Array difference:", c)
    except Exception as e:
        print(f"NumPy test failed: {e}")

if __name__ == "__main__":
    test_numpy()
