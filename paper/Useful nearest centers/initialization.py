import numpy as np
import time

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def find_useful_nearest_centers(points, centers):
    """
    Find useful nearest centers for K-means initialization.

    A center C is useless for a point P if there exists another center Cx where:
    1. dis(P, Cx) < dis(P, C) - Cx is closer to P than C
    2. dis(C, Cx) < dis(P, C) - C and Cx are closer to each other than C is to P

    A center is useful if it is NOT useless.

    Parameters:
    -----------
    points : ndarray of shape (n_points, n_features)
        Data points to find useful centers for
    centers : ndarray of shape (n_centers, n_features)
        Candidate centers

    Returns:
    --------
    useful_centers : list of ndarray
        List of length n_points, where each element is an array of indices
        of useful centers for that point. The closest center is always useful.

    Example:
    --------
    >>> points = np.array([[0, 0], [1, 1], [5, 5]])
    >>> centers = np.array([[0, 0], [1, 0], [2, 2], [6, 6]])
    >>> useful = find_useful_nearest_centers(points, centers)
    >>> print(useful[0])  # Useful centers for point [0, 0]
    [0 1]  # Center 0 is closest, center 1 might be useful too
    """
    n_points = points.shape[0]
    n_centers = centers.shape[0]

    # Compute all pairwise distances
    # P_C_dists[i, j] = distance from point i to center j
    P_C_dists = np.linalg.norm(
        points[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2
    )

    # C_C_dists[i, j] = distance from center i to center j
    C_C_dists = np.linalg.norm(
        centers[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2
    )

    useful_centers = []

    for i in range(n_points):
        # Get distances from point i to all centers
        dists_to_centers = P_C_dists[i]  # Shape: (n_centers,)

        # Initialize all centers as useful
        useful_mask = np.ones(n_centers, dtype=bool)

        for j in range(n_centers):
            dis_P_C = dists_to_centers[j]

            # Find all centers Cx that are closer to P than C
            # (condition 1: dis(P, Cx) < dis(P, C))
            closer_to_P = dists_to_centers < dis_P_C

            # Among those closer centers, check if any satisfy
            # condition 2: dis(C, Cx) < dis(P, C)
            close_to_C = C_C_dists[j, :] < dis_P_C

            # If both conditions are met for any Cx, then C is useless
            if np.any(closer_to_P & close_to_C):
                useful_mask[j] = False

        # Get indices of useful centers for this point
        useful_indices = np.where(useful_mask)[0]
        useful_centers.append(useful_indices)

    return useful_centers


def get_useful_centers_matrix(points, centers):
    """
    Alternative return format as a boolean matrix.

    Returns:
    --------
    useful_matrix : ndarray of shape (n_points, n_centers)
        Boolean matrix where useful_matrix[i, j] = True means
        center j is useful for point i.
    """
    useful_list = find_useful_nearest_centers(points, centers)
    n_points = points.shape[0]
    n_centers = centers.shape[0]

    useful_matrix = np.zeros((n_points, n_centers), dtype=bool)
    for i, indices in enumerate(useful_list):
        useful_matrix[i, indices] = True

    return useful_matrix


def compute_unc_scores(points, centers, exclude_indices=None):
    """
    Compute UNC-based scores for all points.

    For each point P, computes:
    score = (avg_dis / max_dis) * sum_ln_dis

    Where:
    - avg_dis = average distance from P to all its useful nearest centers
    - max_dis = maximum distance from P to all its useful nearest centers
    - sum_ln_dis = sum of ln(distance) from P to all its useful nearest centers

    Parameters:
    -----------
    points : ndarray of shape (n_points, n_features)
        Data points
    centers : ndarray of shape (n_centers, n_features)
        Candidate centers
    exclude_indices : set or None
        Indices of points to exclude from scoring (already selected as centers)

    Returns:
    --------
    scores : ndarray of shape (n_points,)
        UNC-based score for each point. Higher scores indicate points
        that are farther from their useful centers. Points in exclude_indices
        get score of -inf.
    """
    n_points = points.shape[0]

    # Get useful centers for all points
    useful_centers = find_useful_nearest_centers(points, centers)

    # Compute all distances from points to centers
    P_C_dists = np.linalg.norm(
        points[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2
    )

    scores = np.zeros(n_points)

    for i in range(n_points):
        # Skip if this point is already a center
        if exclude_indices and i in exclude_indices:
            scores[i] = -np.inf
            continue

        # Get distances to useful centers for this point
        useful_indices = useful_centers[i]
        useful_distances = P_C_dists[i, useful_indices]

        # Add small epsilon to avoid log(0) when point is exactly at a center
        eps = 1e-10
        useful_distances = np.maximum(useful_distances, eps)

        # Compute components
        avg_dis = np.mean(useful_distances)
        max_dis = np.max(useful_distances)
        sum_ln_dis = np.sum(np.log(useful_distances))

        # Compute score
        scores[i] = (avg_dis / max_dis) * sum_ln_dis

    return scores


def unc_init(points, k, return_indices=False):
    """
    UNC (Useful Nearest Centers) initialization for K-means.

    Algorithm:
    1. First center: select the data point with smallest value on first axis
    2. For each subsequent center:
       - Compute UNC scores for all remaining points
       - Select point with largest score as next center
       - Update centers and useful nearest centers
    3. Repeat until k centers are selected

    Parameters:
    -----------
    points : ndarray of shape (n_points, n_features)
        Data points to cluster
    k : int
        Number of centers to select
    return_indices : bool, default=False
        If True, return indices of selected points. If False, return center coordinates.

    Returns:
    --------
    centers : ndarray of shape (k, n_features) or ndarray of shape (k,)
        If return_indices=False: coordinates of selected centers
        If return_indices=True: indices of selected points as centers
    """
    n_points = points.shape[0]

    if k > n_points:
        raise ValueError(f"k ({k}) cannot be larger than number of points ({n_points})")

    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    # Step 1: Select first center (smallest value on first axis)
    first_center_idx = np.argmin(points[:, 0])
    selected_indices = [first_center_idx]

    # Initialize centers array with the first center
    centers = points[selected_indices].copy()

    # Step 2: Select remaining k-1 centers
    for _ in range(k - 1):
        # Compute UNC scores for all points
        scores = compute_unc_scores(points, centers, exclude_indices=set(selected_indices))

        # Select point with largest score
        next_center_idx = np.argmax(scores)
        selected_indices.append(next_center_idx)

        # Add new center to centers array
        centers = points[selected_indices].copy()

    if return_indices:
        return np.array(selected_indices)
    else:
        return centers


def random_init(points, k, random_state=None):
    """
    Random initialization - randomly select k points as centers.

    Parameters:
    -----------
    points : ndarray of shape (n_points, n_features)
        Data points to cluster
    k : int
        Number of centers to select
    random_state : int or None
        Random seed for reproducibility

    Returns:
    --------
    centers : ndarray of shape (k, n_features)
        Randomly selected centers
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_points = points.shape[0]
    indices = np.random.choice(n_points, size=k, replace=False)
    return points[indices].copy()


def kmeans_plusplus_init(points, k, random_state=None):
    """
    K-means++ initialization.

    Algorithm:
    1. Choose first center uniformly at random
    2. For each subsequent center:
       - Compute distance from each point to nearest existing center
       - Choose next center with probability proportional to squared distance
    3. Repeat until k centers are selected

    Parameters:
    -----------
    points : ndarray of shape (n_points, n_features)
        Data points to cluster
    k : int
        Number of centers to select
    random_state : int or None
        Random seed for reproducibility

    Returns:
    --------
    centers : ndarray of shape (k, n_features)
        Selected centers using k-means++ method
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_points = points.shape[0]

    # Step 1: Choose first center uniformly at random
    first_idx = np.random.randint(n_points)
    centers = [points[first_idx]]

    # Step 2: Choose remaining k-1 centers
    for _ in range(k - 1):
        # Compute distances from each point to nearest center
        distances = np.array([
            min(np.linalg.norm(point - center) for center in centers)
            for point in points
        ])

        # Square the distances for probability distribution
        squared_distances = distances ** 2

        # Avoid division by zero
        if squared_distances.sum() == 0:
            probabilities = np.ones(n_points) / n_points
        else:
            probabilities = squared_distances / squared_distances.sum()

        # Choose next center
        next_idx = np.random.choice(n_points, p=probabilities)
        centers.append(points[next_idx])

    return np.array(centers)


def evaluate_initialization(points, centers):
    """
    Evaluate the quality of initialization using inertia (within-cluster sum of squares).

    Parameters:
    -----------
    points : ndarray of shape (n_points, n_features)
        Data points
    centers : ndarray of shape (k, n_features)
        Selected centers

    Returns:
    --------
    inertia : float
        Sum of squared distances from each point to its nearest center
    """
    # Compute distances from each point to each center
    distances = np.linalg.norm(
        points[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2
    )

    # Find nearest center for each point
    min_distances = np.min(distances, axis=1)

    # Compute inertia
    inertia = np.sum(min_distances ** 2)

    return inertia


def compare_initializations(points, k, n_runs=10, random_state=42):
    """
    Compare UNC, K-means++, and Random initialization methods.

    Parameters:
    -----------
    points : ndarray of shape (n_points, n_features)
        Data points to cluster
    k : int
        Number of clusters
    n_runs : int
        Number of runs for random methods (to get average performance)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    results : dict
        Dictionary containing timing and inertia results for each method
    """
    results = {
        'unc': {'time': [], 'inertia': []},
        'kmeans++': {'time': [], 'inertia': []},
        'random': {'time': [], 'inertia': []}
    }

    print(f"\nComparing initialization methods on {points.shape[0]} points, k={k}")
    print("=" * 70)

    # UNC initialization (single run)
    print("\n1. UNC Initialization:")
    start = time.time()
    unc_centers = unc_init(points, k)
    unc_time = time.time() - start
    unc_inertia = evaluate_initialization(points, unc_centers)
    results['unc']['time'].append(unc_time)
    results['unc']['inertia'].append(unc_inertia)
    print(f"   Time: {unc_time:.4f}s, Inertia: {unc_inertia:.2f}")

    # K-means++ (multiple runs)
    print(f"\n2. K-means++ Initialization ({n_runs} runs):")
    for run in range(n_runs):
        start = time.time()
        kpp_centers = kmeans_plusplus_init(points, k, random_state=random_state + run)
        kpp_time = time.time() - start
        kpp_inertia = evaluate_initialization(points, kpp_centers)
        results['kmeans++']['time'].append(kpp_time)
        results['kmeans++']['inertia'].append(kpp_inertia)

    print(f"   Avg Time: {np.mean(results['kmeans++']['time']):.4f}s")
    print(f"   Avg Inertia: {np.mean(results['kmeans++']['inertia']):.2f} ± {np.std(results['kmeans++']['inertia']):.2f}")
    print(f"   Best Inertia: {np.min(results['kmeans++']['inertia']):.2f}")

    # Random initialization (multiple runs)
    print(f"\n3. Random Initialization ({n_runs} runs):")
    for run in range(n_runs):
        start = time.time()
        rand_centers = random_init(points, k, random_state=random_state + run)
        rand_time = time.time() - start
        rand_inertia = evaluate_initialization(points, rand_centers)
        results['random']['time'].append(rand_time)
        results['random']['inertia'].append(rand_inertia)

    print(f"   Avg Time: {np.mean(results['random']['time']):.4f}s")
    print(f"   Avg Inertia: {np.mean(results['random']['inertia']):.2f} ± {np.std(results['random']['inertia']):.2f}")
    print(f"   Best Inertia: {np.min(results['random']['inertia']):.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  UNC inertia:        {unc_inertia:.2f}")
    print(f"  K-means++ avg:      {np.mean(results['kmeans++']['inertia']):.2f}")
    print(f"  Random avg:         {np.mean(results['random']['inertia']):.2f}")
    print()
    print(f"  UNC vs K-means++ best: {(unc_inertia / np.min(results['kmeans++']['inertia']) - 1) * 100:+.1f}%")
    print(f"  UNC vs Random best:    {(unc_inertia / np.min(results['random']['inertia']) - 1) * 100:+.1f}%")

    return results


def visualize_initialization(points, k, title="Initialization Comparison", figsize=(15, 5)):
    """
    Visualize centers selected by different initialization methods.

    Parameters:
    -----------
    points : ndarray of shape (n_points, 2)
        2D data points to visualize (must be 2D)
    k : int
        Number of clusters
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create visualization.")
        return None, None

    if points.shape[1] != 2:
        print("Visualization only works for 2D data.")
        return None, None

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # UNC initialization
    unc_centers = unc_init(points, k)
    axes[0].scatter(points[:, 0], points[:, 1], alpha=0.3, s=10, c='gray')
    axes[0].scatter(unc_centers[:, 0], unc_centers[:, 1], c='red', s=200, marker='X',
                   edgecolors='black', linewidths=2, label='Centers')
    axes[0].scatter(unc_centers[0, 0], unc_centers[0, 1], c='yellow', s=300, marker='*',
                   edgecolors='black', linewidths=2, label='First center')
    axes[0].set_title('UNC Initialization')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # K-means++ initialization
    kpp_centers = kmeans_plusplus_init(points, k, random_state=42)
    axes[1].scatter(points[:, 0], points[:, 1], alpha=0.3, s=10, c='gray')
    axes[1].scatter(kpp_centers[:, 0], kpp_centers[:, 1], c='blue', s=200, marker='X',
                   edgecolors='black', linewidths=2, label='Centers')
    axes[1].scatter(kpp_centers[0, 0], kpp_centers[0, 1], c='yellow', s=300, marker='*',
                   edgecolors='black', linewidths=2, label='First center')
    axes[1].set_title('K-means++ Initialization')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Random initialization
    rand_centers = random_init(points, k, random_state=42)
    axes[2].scatter(points[:, 0], points[:, 1], alpha=0.3, s=10, c='gray')
    axes[2].scatter(rand_centers[:, 0], rand_centers[:, 1], c='green', s=200, marker='X',
                   edgecolors='black', linewidths=2, label='Centers')
    axes[2].set_title('Random Initialization')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, axes


if __name__ == "__main__":
    # Example usage and test
    np.random.seed(42)

    # Create sample data
    points = np.array([
        [0, 0],
        [1, 1],
        [5, 5],
        [10, 10]
    ])

    centers = np.array([
        [0, 0],
        [0.5, 0.5],
        [2, 2],
        [5, 5],
        [11, 11]
    ])

    print("Points shape:", points.shape)
    print("Centers shape:", centers.shape)
    print()

    useful = find_useful_nearest_centers(points, centers)

    for i, (point, useful_idx) in enumerate(zip(points, useful)):
        print(f"Point {i} at {point}:")
        print(f"  Useful centers: {useful_idx}")
        print(f"  Center locations: {centers[useful_idx]}")

        # Show distances to all centers
        dists = np.linalg.norm(centers - point, axis=1)
        print(f"  Distances to all centers: {dists}")
        print(f"  Closest center: {np.argmin(dists)} (distance: {dists.min():.2f})")
        print()

    # Test matrix format
    print("Useful centers matrix:")
    matrix = get_useful_centers_matrix(points, centers)
    print(matrix)
    print()

    # Test UNC scores
    print("UNC Scores:")
    scores = compute_unc_scores(points, centers)
    for i, (point, score) in enumerate(zip(points, scores)):
        print(f"Point {i} at {point}: score = {score:.4f}")
    print()
    print("Point with highest score (farthest from useful centers):", np.argmax(scores))
    print()

    # Test UNC initialization algorithm
    print("=" * 60)
    print("Testing UNC Initialization Algorithm")
    print("=" * 60)

    # Create a larger dataset for testing
    np.random.seed(42)
    test_points = np.random.randn(50, 2) * 10

    print(f"Dataset: {test_points.shape[0]} points in {test_points.shape[1]}D space")
    print()

    # Test with different k values
    for k in [3, 5, 8]:
        print(f"Selecting k={k} centers using UNC initialization:")
        selected_centers = unc_init(test_points, k)
        selected_indices = unc_init(test_points, k, return_indices=True)

        print(f"  Selected indices: {selected_indices}")
        print(f"  First center (min on axis 0): index {selected_indices[0]}, point {selected_centers[0]}")
        print(f"  Centers shape: {selected_centers.shape}")
        print()

    print("UNC initialization test completed successfully!")
    print()

    # Comparison with other initialization methods
    print("\n" + "=" * 70)
    print("COMPARING INITIALIZATION METHODS")
    print("=" * 70)

    # Create a more realistic dataset with clusters
    np.random.seed(42)

    # Generate 3 clusters
    cluster1 = np.random.randn(100, 2) * 2 + np.array([0, 0])
    cluster2 = np.random.randn(100, 2) * 2 + np.array([15, 15])
    cluster3 = np.random.randn(100, 2) * 2 + np.array([15, -15])
    comparison_points = np.vstack([cluster1, cluster2, cluster3])

    print(f"\nTest dataset: {comparison_points.shape[0]} points with 3 natural clusters")

    # Compare methods for k=3
    results = compare_initializations(comparison_points, k=3, n_runs=10, random_state=42)

    # Visualize the initializations
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating visualization...")
        fig, axes = visualize_initialization(comparison_points, k=3,
                                            title="Initialization Methods Comparison (k=3)")
        if fig is not None:
            plt.savefig('initialization_comparison.png', dpi=150, bbox_inches='tight')
            print("Visualization saved to 'initialization_comparison.png'")
            # plt.show()  # Uncomment to display interactively
    else:
        print("\nMatplotlib not available. Skipping visualization.")

    print("\n" + "=" * 70)
    print("NOTE: UNC initialization is deterministic and designed to work well")
    print("with the full K-means algorithm. Initial inertia may be higher than")
    print("K-means++, but can lead to better final clustering results after")
    print("K-means iterations. The paper suggests UNC provides more robust")
    print("initialization across different datasets.")
    print("=" * 70)
    print("\nAll tests completed successfully!")
