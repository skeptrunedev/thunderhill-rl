# Human rider camera

20 September 2026. The human rider camera now follows the motorcycle's rendered pose directly. Previously it used the chase camera's world position smoothing; steady motion therefore caused the viewpoint to trail the motorcycle, rather than stay at the rider.

The eye anchor is inside the original helmet mesh at local coordinates (0, 1.51, -0.30) metres. It follows pitch and lean. Gaze follows the road tangent with a 0.25 radian downward framing angle and 22 percent roll response. Vertical field of view is 90 degrees; switching to chase restores 64 degrees and its existing position smoothing. These are provisional presentation choices, not measurements of the reference camera or human head stabilization. The rendered rider remains excluded to prevent helmet occlusion.

The dedicated agent observation camera remains at its existing versioned pose and intrinsics. Human presentation settings do not silently change policy observations. No physics or control inputs changed.

`tests/test_rider_camera.gd` checks fixed eye attachment after large position changes, sloped road samples, mirrored lean and different frame intervals, plus visibility and field of view when switching modes. Actual Vulkan renders were inspected upright and at 0.4 radians lean. The dashboard remains visible near the bottom of the frame, with more road in view. Artifact paths are `artifacts/helmet-view-framing.png` and `artifacts/rider-camera-lean.png`.

Camera motion and viewing comfort still need human riding review. This correction does not address the provisional motorcycle mesh, missing rider arm presentation, or the remaining track and dynamics realism work.
