"""
Maps a viewport orientation (roll, pitch, yaw) to binary patch-relevance labels
over a grid that splits an equirectangular video frame into N patches.

Convention (matches preprocess.py's Euler angle output):
- yaw   in [-180, 180], wraps around (yaw=180 and yaw=-180 are the same meridian)
- pitch in [-90, 90], no wrap-around (poles are boundaries)
- roll is ignored: it rotates the viewing frustum around the gaze axis but does not
  move the gaze center point, so it does not affect which patch the viewport falls in.
"""
import numpy as np


def yaw_pitch_to_grid_cell(yaw, pitch, grid_rows, grid_cols):
    """
    Map yaw/pitch (in degrees) to a (row, col) grid cell index.

    :param yaw: yaw angle(s) in degrees, range [-180, 180]
    :param pitch: pitch angle(s) in degrees, range [-90, 90]
    :param grid_rows: number of grid rows (latitude bins)
    :param grid_cols: number of grid columns (longitude bins)
    :return: (row, col) as int or int array
    """
    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)

    # normalize yaw into [0, 360) before binning so -180 and 180 land on the same edge
    x = (yaw + 180.0) % 360.0
    col = np.floor(x / 360.0 * grid_cols).astype(np.int64)
    col = np.clip(col, 0, grid_cols - 1)

    # pitch: 90 (top) -> row 0, -90 (bottom) -> row grid_rows - 1
    y = (90.0 - pitch) / 180.0 * grid_rows
    row = np.floor(y).astype(np.int64)
    row = np.clip(row, 0, grid_rows - 1)

    return row, col


def grid_cell_to_patch_index(row, col, grid_cols):
    return row * grid_cols + col


def viewport_to_patch_labels(yaw, pitch, grid_rows, grid_cols, include_neighbors=True):
    """
    Build a binary label vector of length grid_rows*grid_cols for a single viewport sample.
    The patch containing (yaw, pitch) is positive; if include_neighbors, its 4-connected
    neighbors (up/down/left/right, with column wrap-around) are also positive.

    :return: numpy array of shape (grid_rows*grid_cols,), dtype float32, values in {0., 1.}
    """
    n_patches = grid_rows * grid_cols
    labels = np.zeros(n_patches, dtype=np.float32)

    row, col = yaw_pitch_to_grid_cell(yaw, pitch, grid_rows, grid_cols)
    row, col = int(row), int(col)
    labels[grid_cell_to_patch_index(row, col, grid_cols)] = 1.0

    if include_neighbors:
        neighbors = [
            (row - 1, col),
            (row + 1, col),
            (row, (col - 1) % grid_cols),  # wrap around left
            (row, (col + 1) % grid_cols),  # wrap around right
        ]
        for r, c in neighbors:
            if 0 <= r < grid_rows:  # no wrap-around on pitch/rows
                labels[grid_cell_to_patch_index(r, c, grid_cols)] = 1.0

    return labels


def viewport_sequence_to_patch_labels(future_viewports, grid_rows, grid_cols, include_neighbors=True):
    """
    Union of per-timestep patch labels over a future viewport window, i.e. a patch is
    positive if it is relevant at ANY timestep in the future window.

    :param future_viewports: array-like of shape (T, 3) with columns (roll, pitch, yaw),
        matching the column order produced by preprocess.py / load_dataset.py.
    :return: numpy array of shape (grid_rows*grid_cols,), dtype float32
    """
    future_viewports = np.asarray(future_viewports, dtype=np.float64)
    n_patches = grid_rows * grid_cols
    labels = np.zeros(n_patches, dtype=np.float32)
    for row in future_viewports:
        _, pitch, yaw = row[0], row[1], row[2]
        labels = np.maximum(
            labels,
            viewport_to_patch_labels(yaw, pitch, grid_rows, grid_cols, include_neighbors)
        )
    return labels


if __name__ == '__main__':
    # sanity checks (no GPU/torch needed)
    grid_rows, grid_cols = 4, 4

    # center of the frame (yaw=0, pitch=0) should land near the middle patches
    row, col = yaw_pitch_to_grid_cell(0.0, 0.0, grid_rows, grid_cols)
    assert (row, col) == (2, 2), f'expected (2,2), got {(row, col)}'

    # yaw wrap-around: -180 and 180 should map to the same column
    row1, col1 = yaw_pitch_to_grid_cell(180.0, 0.0, grid_rows, grid_cols)
    row2, col2 = yaw_pitch_to_grid_cell(-180.0, 0.0, grid_rows, grid_cols)
    assert col1 == col2 == 0, f'wrap-around column mismatch: {col1} vs {col2}'

    # top/bottom pitch extremes should clamp to row 0 / row grid_rows-1
    row_top, _ = yaw_pitch_to_grid_cell(0.0, 90.0, grid_rows, grid_cols)
    row_bottom, _ = yaw_pitch_to_grid_cell(0.0, -90.0, grid_rows, grid_cols)
    assert row_top == 0 and row_bottom == grid_rows - 1

    # neighbor wrap-around at the yaw seam (col=0 neighbor to the left should be col=grid_cols-1)
    labels = viewport_to_patch_labels(yaw=-179.0, pitch=0.0, grid_rows=grid_rows, grid_cols=grid_cols,
                                       include_neighbors=True)
    assert labels[grid_cell_to_patch_index(2, grid_cols - 1, grid_cols)] == 1.0, \
        'left-neighbor wrap-around across the yaw seam failed'

    # sequence union: a viewport sweeping across two different patches should positive-label both centers
    seq = np.array([
        [0.0, 0.0, 0.0],     # roll, pitch, yaw -> center patch (2,2)
        [0.0, 0.0, -179.0],  # -> seam patch (2,0)
    ])
    seq_labels = viewport_sequence_to_patch_labels(seq, grid_rows, grid_cols, include_neighbors=False)
    assert seq_labels[grid_cell_to_patch_index(2, 2, grid_cols)] == 1.0
    assert seq_labels[grid_cell_to_patch_index(2, 0, grid_cols)] == 1.0
    assert seq_labels.sum() == 2.0

    print('All patch_labeling self-tests passed.')
