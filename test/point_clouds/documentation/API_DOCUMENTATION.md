# Point Cloud Registration API Documentation

## test_registration.py - Complete Function Reference

### Core Functions

#### `load_point_cloud(file_path)`
**Purpose**: Load point cloud from PLY file with validation
**Parameters**:
- `file_path` (str/Path): Path to PLY file
**Returns**: `o3d.geometry.PointCloud`
**Raises**: `ValueError` if file is empty or invalid

#### `generate_random_transformation(rotation_range=45, translation_range=0.5)`
**Purpose**: Generate random 4x4 transformation matrix
**Parameters**:
- `rotation_range` (float): Max rotation angle in degrees (default: 45)
- `translation_range` (float): Max translation distance (default: 0.5)
**Returns**: 
- `transformation_matrix` (4x4 np.array): Homogeneous transformation
- `rotation_angles` (tuple): (rx, ry, rz) in degrees
- `translation` (tuple): (tx, ty, tz) in meters

#### `apply_transformation(pcd, transformation_matrix)`
**Purpose**: Apply transformation to point cloud (creates copy)
**Parameters**:
- `pcd` (o3d.geometry.PointCloud): Source point cloud
- `transformation_matrix` (4x4 np.array): Transformation to apply
**Returns**: `o3d.geometry.PointCloud` (transformed copy)

### Registration Algorithms

#### `test_icp_registration(source_pcd, target_pcd, threshold=0.1)`
**Purpose**: Standard ICP registration with multiple initializations
**Parameters**:
- `source_pcd` (o3d.geometry.PointCloud): Source point cloud
- `target_pcd` (o3d.geometry.PointCloud): Target point cloud
- `threshold` (float): Distance threshold for correspondences
**Returns**: Dict with keys:
- `transformation` (4x4 np.array): Estimated transformation
- `fitness` (float): Fraction of inlier correspondences
- `inlier_rmse` (float): RMSE of inlier points
- `computation_time` (float): Runtime in seconds

**Algorithm Details**:
- Tries identity matrix and 5 random initializations
- Uses point-to-point ICP estimation
- Falls back to simple ICP if all attempts fail

#### `test_robust_icp_registration(source_pcd, target_pcd)`
**Purpose**: Robust ICP using RANSAC + ICP
**Parameters**:
- `source_pcd` (o3d.geometry.PointCloud): Source point cloud
- `target_pcd` (o3d.geometry.PointCloud): Target point cloud
**Returns**: Same format as `test_icp_registration`

**Algorithm Details**:
- Estimates normals for both point clouds
- Computes FPFH features
- Uses RANSAC for initial alignment
- Refines with ICP

#### `test_global_registration(source_pcd, target_pcd)`
**Purpose**: Feature-based global registration
**Parameters**:
- `source_pcd` (o3d.geometry.PointCloud): Source point cloud
- `target_pcd` (o3d.geometry.PointCloud): Target point cloud
**Returns**: Same format as `test_icp_registration`

**Algorithm Details**:
- Estimates normals and computes FPFH features
- Uses Fast Global Registration based on feature matching
- No ICP refinement (pure feature-based)

#### `create_bounding_box_registration(source_pcd, target_pcd, box_size=1.0)`
**Purpose**: Simple center-based bounding box alignment
**Parameters**:
- `source_pcd` (o3d.geometry.PointCloud): Source point cloud
- `target_pcd` (o3d.geometry.PointCloud): Target point cloud
- `box_size` (float): Size of bounding box (unused in current implementation)
**Returns**: Same format as `test_icp_registration`

**Algorithm Details**:
- Computes axis-aligned bounding boxes
- Aligns centers
- No rotation handling

#### `create_pca_bounding_box_registration(source_pcd, target_pcd, box_size=1.0)`
**Purpose**: PCA-based bounding box alignment
**Parameters**:
- `source_pcd` (o3d.geometry.PointCloud): Source point cloud
- `target_pcd` (o3d.geometry.PointCloud): Target point cloud
- `box_size` (float): Size of bounding box (unused in current implementation)
**Returns**: Same format as `test_icp_registration`

**Algorithm Details**:
- Centers both point clouds
- Computes PCA for both
- Aligns principal axes
- Handles rotations through PCA alignment

#### `create_hybrid_bounding_box_registration(source_pcd, target_pcd, box_size=1.0)`
**Purpose**: PCA bounding box + ICP refinement
**Parameters**:
- `source_pcd` (o3d.geometry.PointCloud): Source point cloud
- `target_pcd` (o3d.geometry.PointCloud): Target point cloud
- `box_size` (float): Size of bounding box (unused in current implementation)
**Returns**: Same format as `test_icp_registration`

**Algorithm Details**:
- Uses PCA bounding box for initial alignment
- Refines with ICP using threshold=0.1

### Evaluation Functions

#### `evaluate_transformation(estimated_transform, ground_truth_transform, tolerance=1e-3, consider_symmetry=True, symmetry_axes=None)`
**Purpose**: Evaluate registration quality with symmetry consideration
**Parameters**:
- `estimated_transform` (4x4 np.array): Algorithm result
- `ground_truth_transform` (4x4 np.array): True transformation
- `tolerance` (float): Error tolerance for "correct" classification
- `consider_symmetry` (bool): Whether to check symmetric solutions
- `symmetry_axes` (list): List of symmetry transformation matrices
**Returns**: Dict with keys:
- `rotation_error_degrees` (float): Angular error in degrees
- `translation_error` (float): Translation error in meters
- `is_correct` (bool): Whether registration meets tolerance
- `is_symmetric` (bool): Whether best match was symmetric
- `symmetry_index` (int): Index of best symmetric match (-1 if direct)
- `symmetric_solutions` (list): Details of all symmetric evaluations

#### `get_t_shape_symmetries()`
**Purpose**: Define T-shape symmetry transformations
**Returns**: List of 4x4 transformation matrices:
- Identity (no rotation)
- Z-180° rotation
- X-180° rotation  
- Y-180° rotation

#### `analyze_symmetry_impact(source_pcd, target_pcd, estimated_transform, ground_truth_transform)`
**Purpose**: Detailed symmetry analysis for debugging
**Parameters**:
- `source_pcd` (o3d.geometry.PointCloud): Source point cloud
- `target_pcd` (o3d.geometry.PointCloud): Target point cloud
- `estimated_transform` (4x4 np.array): Algorithm result
- `ground_truth_transform` (4x4 np.array): True transformation
**Returns**: None (prints detailed analysis)

### Visualization Functions

#### `create_registration_summary_visualization(source_pcd, target_pcd, estimated_pcd, ground_truth_transform, estimated_transform, rotation_error, translation_error, title)`
**Purpose**: Create main 3D visualization with all point clouds
**Parameters**:
- `source_pcd` (o3d.geometry.PointCloud): Source point cloud
- `target_pcd` (o3d.geometry.PointCloud): Target point cloud
- `estimated_pcd` (o3d.geometry.PointCloud): Registration result
- `ground_truth_transform` (4x4 np.array): True transformation
- `estimated_transform` (4x4 np.array): Algorithm result
- `rotation_error` (float): Rotation error in degrees
- `translation_error` (float): Translation error in meters
- `title` (str): Window title
**Returns**: None (displays visualization)

**Visualization Elements**:
- Red point cloud: Source
- Green point cloud: Target
- Blue point cloud: Estimated
- Yellow line: Ground truth transformation
- Magenta line: Estimated transformation
- Coordinate frames at each center
- Console output with quality assessment

#### `create_side_by_side_visualization(source_pcd, target_pcd, estimated_pcd, title)`
**Purpose**: Side-by-side comparison when point clouds overlap
**Parameters**:
- `source_pcd` (o3d.geometry.PointCloud): Source point cloud
- `target_pcd` (o3d.geometry.PointCloud): Target point cloud
- `estimated_pcd` (o3d.geometry.PointCloud): Registration result
- `title` (str): Window title
**Returns**: None (displays visualization)

**Visualization Elements**:
- Left: Source (red)
- Middle: Target (green)
- Right: Estimated (blue)
- Coordinate frames and center markers
- Automatic spacing calculation

### Utility Functions

#### `save_results(results, file_path)`
**Purpose**: Save results to JSON file with proper serialization
**Parameters**:
- `results` (dict): Results dictionary
- `file_path` (Path): Output file path
**Returns**: None

**Handles**:
- NumPy arrays → Python lists
- Open3D Vector types → Python lists
- Recursive conversion of nested structures

#### `convert_numpy(obj)`
**Purpose**: Recursively convert NumPy/Open3D objects to JSON-serializable types
**Parameters**:
- `obj`: Object to convert
**Returns**: Converted object

### Main Function

#### `main(test_single_algorithm=None, force_side_by_side=False)`
**Purpose**: Main testing pipeline
**Parameters**:
- `test_single_algorithm` (str, optional): Algorithm name to test exclusively
- `force_side_by_side` (bool): Force side-by-side visualization
**Returns**: None

**Pipeline**:
1. Load point cloud from `t_pcd.ply`
2. Generate random transformation
3. Apply transformation to create target
4. Test selected algorithms
5. Evaluate results with symmetry analysis
6. Create visualizations
7. Print summary

**Command Line Arguments**:
- `--side-by-side` or `-s`: Force side-by-side visualization
- Algorithm name: Test specific algorithm only

## Configuration Constants

```python
# Default transformation ranges
ROTATION_RANGE = 15  # degrees
TRANSLATION_RANGE = 0.2  # meters

# ICP parameters
ICP_THRESHOLD = 2.0  # Distance threshold
ICP_MAX_ITERATIONS = 100

# Evaluation parameters
EVALUATION_TOLERANCE = 1e-3
SYMMETRY_IMPROVEMENT_THRESHOLD = 10.0  # degrees + meters

# Visualization parameters
POINT_SIZE = 3.0
LINE_WIDTH = 4.0
```

## Error Handling

### Common Exceptions
- `ValueError`: Empty or invalid point cloud files
- `FileNotFoundError`: Missing PLY file
- `AttributeError`: Open3D object attribute issues
- `TypeError`: JSON serialization errors

### Fallback Mechanisms
- ICP: Falls back to simple ICP if multiple initializations fail
- JSON: Graceful handling of serialization errors
- Visualization: Automatic side-by-side when point clouds overlap

## Performance Characteristics

### Algorithm Complexity
- **ICP**: O(n²) per iteration, typically 10-100 iterations
- **Robust ICP**: O(n log n) for FPFH + O(n²) for ICP
- **Global Registration**: O(n log n) for FPFH + O(n) for RANSAC
- **Bounding Box Methods**: O(n) for PCA computation

### Memory Usage
- Point clouds: ~10,000 points × 3 coordinates × 4 bytes = ~120KB each
- FPFH features: ~10,000 points × 33 features × 4 bytes = ~1.3MB each
- Total memory: ~5-10MB for typical usage

### Typical Performance
- **ICP**: 0.5-2 seconds
- **Robust ICP**: 2-5 seconds  
- **Global Registration**: 1-3 seconds
- **Bounding Box Methods**: 0.1-0.5 seconds
