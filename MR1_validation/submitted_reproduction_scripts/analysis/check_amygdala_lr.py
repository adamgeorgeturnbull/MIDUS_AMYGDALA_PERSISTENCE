"""
check_amygdala_lr.py

Confirms L/R amygdala mask assignment by reporting MNI centroids.
In MNI space: left hemisphere = negative x, right = positive x.

Run on Sherlock after loading FSL/nilearn modules.
"""
from nilearn import image, datasets
import numpy as np

atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')
labels = atlas.labels
atlas_img = atlas.filename

for hemi, label in [('L', 'Left Amygdala'), ('R', 'Right Amygdala')]:
    idx = labels.index(label)
    mask = image.math_img(f"img == {idx}", img=atlas_img)
    coords = np.array(np.where(mask.get_fdata() > 0)).T
    affine = mask.affine
    mni = np.dot(affine[:3, :3], coords.T).T + affine[:3, 3]
    centroid = mni.mean(axis=0)
    print(f"{hemi} ({label}): MNI centroid x={centroid[0]:.1f}, y={centroid[1]:.1f}, z={centroid[2]:.1f}  (n_vox={len(coords)})")
    print(f"  x range: {mni[:,0].min():.1f} to {mni[:,0].max():.1f}  (should be {'negative' if hemi=='L' else 'positive'})")
