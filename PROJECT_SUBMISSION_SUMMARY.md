 Vision Aided Navigation 2026 - Project Submission Summary

## Project Overview

**Goal**: Estimate the trajectory of a vehicle from a video captured with an onboard stereo camera using a multi-stage algorithm pipeline.

**Course**: Vision Aided Navigation 2026

**Dataset**: KITTI odometry sequence 00 with stereo images and ground truth poses

---

## Project Structure

The system implements a complete Visual SLAM pipeline across **7 exercises**, each building upon the previous:

1. **Exercise 1**: ORB feature detection and stereo matching
2. **Exercise 2**: Stereo rectification and triangulation
3. **Exercise 3**: Temporal tracking and PnP-based motion estimation
4. **Exercise 4**: Tracking database construction and statistics
5. **Exercise 5**: Bundle adjustment and keyframe optimization
6. **Exercise 6**: Pose graph construction and optimization
7. **Exercise 7**: Loop closure detection and graph refinement

---

## Report Requirements

### 1. Introduction and Overview
- Provide brief background on why trajectory estimation is important
- Explain the problem context and significance

### 2. Code Organization & Documentation

**Must specify exact locations (file name, function name, line number) for:**

- **Triangulation** - Reconstructing 3D points from stereo observations
- **RANSAC** - Robust pose estimation with outlier rejection
- **PnP Trajectory Calculation** - Camera pose estimation and trajectory tracking
- **Database Definition** - TrackingDB structure and core data organization
- **Adding Frames to Database** - Temporal tracking integration
- **Single Bundle Creation** - Bundle adjustment for a single window
- **Relative Transformation & Covariance Extraction** - Extracting pose uncertainty from bundle results
- **Pose Graph Building** - Factor graph construction from keyframe constraints
- **Loop Closure Detection & Factor Creation** - Finding and verifying loop closures

**Code Quality Requirements:**
- Must be readable and well-documented
- Clear function organization
- Proper section markers for navigability

### 3. Performance Analysis (ביצועים חקר)

#### Mandatory Tracking Statistics
- Total number of tracks
- Number of frames
- Mean track length
- Mean number of frame links
- Graph: Number of matches per frame
- Graph: Percentage of inliers per frame

#### Suggested Analysis Graphs (substantial subset required)

1. **Connectivity Graph**
   - For each frame, number of tracks with links to the next frame
   - Shows temporal feature continuity

2. **Track Length Histogram**
   - Distribution of how long features are tracked
   - Use log scale if appropriate

3. **Trajectory Comparison (Bird's Eye View)**
   - PnP estimation trajectory
   - Bundle adjustment estimation trajectory
   - Pose Graph (with loop closure) trajectory
   - Ground truth trajectory (in green)
   - All overlaid for comparison

4. **Optimization Error**
   - X-axis: Keyframes
   - Y-axis: Mean factor error (total error / #factors)
   - Two lines: Initial error and optimized error
   - One graph per bundle window

5. **Median Projection Error (Bundle)**
   - X-axis: Keyframes
   - Y-axis: Median projection error
   - Two lines: Before and after optimization

6. **Projection Error vs Track Distance**
   - X-axis: Distance from reference frame
   - Y-axis: Median reprojection error
   - For both PnP and Bundle estimation
   - Can use representative subset of tracks

7. **Absolute PnP Estimation Error**
   - Location errors: X, Y, Z axis errors, Total location norm (m)
   - Rotation error (deg)

8. **Absolute Pose Graph Error (without loop closure)**
   - Location errors: X, Y, Z axis errors, Total location norm (m)
   - Rotation error (deg)

9. **Absolute Pose Graph Error (with loop closure)**
   - Location errors: X, Y, Z axis errors, Total location norm (m)
   - Rotation error (deg)

10. **Relative Error Analysis**
    - Error between consecutive keyframes
    - Compare estimated vs ground truth relative poses
    - For both Bundle and PnP estimation
    - Location error norm (m)
    - Angle error (deg)
    - Check for ground truth accuracy issues

11. **Relative Error Over Sub-sections**
    - Measure error on sequences of different lengths: 100, 400, 800 frames
    - Location error norm (as error%: m/m)
    - Angle error (as deg/m)
    - Calculate average error for each metric

12. **Loop Closure Statistics**
    - Number of matches per successful loop closure frame
    - Inlier percentage per successful loop closure frame

13. **Uncertainty Size vs Keyframe (without loop closure)**
    - Location uncertainty
    - Angle uncertainty

14. **Uncertainty Size vs Keyframe (with loop closure)**
    - Location uncertainty
    - Angle uncertainty
    - Show impact of loop closures on reducing drift

#### Graph Requirements
- All graphs must be numbered (Figure 1, Figure 2, etc.)
- Every axis must have units labeled (m, deg, pixels, etc.)
- Specify exact code location for each figure (file and line number)
- Use appropriate scales (log scale where meaningful)
- Zoom in on important data, drop irrelevant parts
- Use colors effectively
- Overlay data when appropriate for comparison
- Use same scale for comparable graphs

### 4. Discussion and Conclusions

**Include:**
- Summary of results
- Identify weak points in the system
- Point out unrealistic assumptions
- Suggest improvements
- Discuss potential future research directions

**If extra work was done:**
- Describe what was implemented beyond requirements
- Explain motivation
- Present results

**Report Constraints:**
- **Maximum 20 pages** - be concise
- All pages and figures numbered
- Exact code location specified for each figure

---

## Oral Discussion Requirements

### Purpose
30-45 minute video call to explain and defend your work, demonstrating:
- Understanding of the project
- Knowledge of the codebase
- Grasp of course material as presented in class

### Scheduling Requirements

Include in submission:
1. **Three Possible Dates**
   - Propose 3 distinct dates
   - All at 18:00
   - Within a 3-week window starting 2 weeks after submission

2. **Contact Email**
   - Reliable email for scheduling and video coordination

### Preparation Expectations

Be prepared to discuss:
- **Design Rationale**: Why you chose your implementation approach
- **Implementation Details**: How the final system works
- **Codebase Navigation**: Demonstrate clear understanding of code organization and flow
- **Course Material**: Show understanding of concepts as taught in class

**Key Topics to Prepare:**
- Triangulation algorithms and their properties
- RANSAC for robust pose estimation
- PnP algorithm and trajectory estimation
- Feature tracking and database management
- Bundle adjustment optimization
- Pose graph construction and optimization
- Loop closure detection and verification

---

## Error Metrics & Definitions

### Relative Error Calculation (from Appendix)

Used to measure accuracy of relative pose estimates between two poses.

#### Rotation Comparison (Rodrigues Formula)
```python
rvec, _ = cv2.Rodrigues(R)
angle_degrees = numpy.linalg.norm(rvec) * 180 / numpy.pi
```

#### Location Error
```
|error(a, b)|_loc / total_distance(a, b)
```
Where:
- `|error(a, b)|_loc` = location difference between estimated and true movement
- `total_distance(a, b)` = sum of all ground truth distances between frames a and b

#### Angle Error
```
|error(a, b)|_ang / total_distance(a, b)
```

#### Sub-section Error (KITTI Style)
- Measure error on sequences of length 100, 400, 800 frames
- Error per meter: `error_distance / trajectory_length`
- Error per meter: `error_angle / trajectory_length`

---

## Key Implementation Stages

### Stage 1: Feature Extraction & Matching
- ORB feature detection on stereo pairs
- Descriptor matching and filtering (Lowe's ratio test)

### Stage 2: Stereo Rectification & Triangulation
- Rectified stereo validation
- Linear SVD triangulation
- OpenCV triangulation comparison

### Stage 3: Temporal Tracking & PnP
- Feature matching across consecutive frames
- 4-point PnP pose estimation
- PnP-RANSAC with dynamic stopping
- Frame-by-frame trajectory tracking

### Stage 4: Tracking Database
- Long-term feature track management
- Track statistics and connectivity analysis
- Inlier percentage tracking

### Stage 5: Bundle Adjustment
- GTSAM-based optimization
- Windowed bundle adjustment
- Keyframe selection strategy
- Covariance computation via Schur complements

### Stage 6: Pose Graph Optimization
- Factor graph from relative pose constraints
- Pose graph optimization
- Marginal covariance extraction
- Uncertainty analysis

### Stage 7: Loop Closure Detection
- Candidate detection via Mahalanobis distance
- Visual verification via AKAZE + Fundamental matrix RANSAC
- Relative pose estimation for verified loops
- Graph re-optimization with loop constraints

---

## Evaluation Criteria

### Code Quality
✓ Readable and well-documented
✓ Clear function organization
✓ Proper location specification for key stages

### Analysis Completeness
✓ Substantial subset of suggested graphs
✓ All mandatory metrics included
✓ Proper axis labels and scaling
✓ Code locations for all figures

### Report Quality
✓ Clear explanation of algorithms
✓ Coherent performance analysis
✓ Constructive criticism and future directions
✓ Under 20 pages (concise)

### Oral Discussion Readiness
✓ Can navigate codebase effectively
✓ Can explain design rationale
✓ Can discuss course material comprehensively
✓ Can defend implementation choices

---

## Notes for Success

1. **Understand the Pipeline**: Each stage builds on the previous. Know how they connect.

2. **Code Navigation**: Be able to quickly find and explain any function marked with the required stage labels.

3. **Graphs Tell Stories**: Every graph should convey something meaningful about system performance.

4. **Error Analysis**: Understand why errors increase/decrease at different stages.

5. **Trade-offs**: Be ready to discuss trade-offs between accuracy, computation, and robustness.

6. **Ground Truth**: Know where ground truth comes from and potential issues with it.

7. **Loop Closures**: Understand how they reduce drift and their impact on uncertainty.

---

## Timeline

- **Submission Date**: [Your date]
- **Oral Discussion Window**: 2-3 weeks after submission
- **Discussion Time**: 18:00 (fixed)
- **Duration**: 30-45 minutes
