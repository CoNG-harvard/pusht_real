# Point Cloud Registration - Quick Reference

## 🚀 Quick Start

```bash
# Activate environment
conda activate pusht_real

# Test all algorithms
python test/point_clouds/test_registration.py

# Test specific algorithm
python test/point_clouds/test_registration.py ICP
python test/point_clouds/test_registration.py Robust_ICP
python test/point_clouds/test_registration.py PCA_Bounding_Box

# Force side-by-side view
python test/point_clouds/test_registration.py --side-by-side
```

## 📊 Available Algorithms

| Algorithm | Best For | Speed | Accuracy | Notes |
|-----------|----------|-------|----------|-------|
| **ICP** | Small transformations | Fast | High | Threshold=2.0 |
| **Robust ICP** | Large misalignments | Medium | High | RANSAC + ICP |
| **Global Registration** | Complex scenarios | Medium | Medium | Feature-based |
| **Bounding Box** | Simple alignment | Very Fast | Low | Center-only |
| **PCA Bounding Box** | Rotated objects | Fast | Medium | PCA alignment |
| **Hybrid** | Best of both | Medium | High | PCA + ICP |

## 🎯 Visualization Guide

### Colors
- 🔴 **Red**: Source (original point cloud)
- 🟢 **Green**: Target (ground truth transformed)  
- 🔵 **Blue**: Estimated (registration result)

### Lines
- 🟡 **Yellow**: Ground truth transformation
- 🟣 **Magenta**: Estimated transformation

### Success Criteria
- ✅ **Good**: Blue overlaps with Green
- ❌ **Poor**: Blue far from Green
- ⚠️ **Check**: Use side-by-side view if overlapping

## 🔧 Key Parameters

### Transformation Range
```python
rotation_range=15      # degrees (smaller = better success)
translation_range=0.2  # meters (smaller = better success)
```

### ICP Threshold
```python
threshold=2.0  # Distance threshold (critical for success)
```

### Evaluation
```python
tolerance=1e-3  # Error tolerance for "correct" classification
```

## 🎭 T-Shape Symmetry

For T-shaped objects, 4 orientations are valid:
- **Identity**: Direct match (0°)
- **Z-180°**: Vertical flip (180° around Z)
- **X-180°**: Horizontal flip (180° around X)
- **Y-180°**: Horizontal flip (180° around Y)

## 🐛 Troubleshooting

### "Red point cloud not showing up"
```bash
python test/point_clouds/test_registration.py --side-by-side
```

### "Algorithm returns identity"
- Check threshold parameters
- Try smaller transformation ranges
- Verify point cloud has features

### "High errors but looks correct"
- Check symmetry analysis
- May be valid flipped orientation
- Look at relative performance ratios

## 📈 Performance Tips

1. **For ICP**: Use threshold=2.0, small transformation ranges
2. **For symmetric objects**: Enable symmetry analysis
3. **For visualization**: Use side-by-side when overlapping
4. **For debugging**: Run diagnostic script

## 🔍 Diagnostic Commands

```bash
# Run diagnostics
python test/point_clouds/diagnose_registration.py

# Show help
python test/point_clouds/visualization_demo.py help

# Demonstrate symmetry
python test/point_clouds/symmetry_demo.py
```

## 📁 File Structure

```
test/point_clouds/
├── test_registration.py      # Main script
├── diagnose_registration.py  # Diagnostics
├── symmetry_demo.py         # Symmetry demo
├── visualization_demo.py    # Help
├── t_pcd.ply               # Test data
└── documentation/          # Documentation folder
    ├── README.md           # Full documentation
    ├── API_DOCUMENTATION.md # Technical reference
    ├── QUICK_REFERENCE.md  # This file
    └── LESSONS_LEARNED.md  # Key insights
```

## 🎯 Expected Results

### Good Performance
- **Fitness**: > 0.3
- **Rotation Error**: < 5°
- **Translation Error**: < 0.1m
- **Time**: < 5 seconds

### Excellent Performance  
- **Fitness**: > 0.8
- **Rotation Error**: < 1°
- **Translation Error**: < 0.01m
- **Time**: < 2 seconds

## ⚡ Quick Commands

```bash
# Test with small transformation
python test/point_clouds/test_registration.py ICP

# Test with side-by-side view
python test/point_clouds/test_registration.py Robust_ICP --side-by-side

# Run diagnostics
python test/point_clouds/diagnose_registration.py

# Show all options
python test/point_clouds/visualization_demo.py help
```
