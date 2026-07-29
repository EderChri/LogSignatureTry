"""channel_adapt.py — resolve a raw-channel-count mismatch between a finetune
dataset and its cross-dataset pretrain checkpoint (e.g. pretrain on capture24's
3-channel wrist accelerometer, finetune on Opportunity's 113 channels).

Two mutually exclusive strategies, selected via --channel_adapt (wired up in
scripts/run_finetune.py):

  drop — keep the K=pretrain_num_feature most-similar finetune channels,
         dropping the rest, so the pretrained input projection
         (branches.0.proj) transfers onto the kept channels with no
         reinitialisation. Channel order is chosen so the kept channel at
         position j is the one assigned to pretrain channel j.
  pca  — PCA-reduce the finetune channels to K=pretrain_num_feature
         components. Reuses the existing --pca_components machinery in
         src/dataloader.py; this module is not involved.

Channel similarity (used by 'drop') is metadata-only — no signal statistics.
Two channels can only match if they have identical modality (accel/gyro/...)
AND identical axis (x/y/z): a back_x channel never matches a back_y channel,
and an accelerometer channel never matches a gyroscope channel, even at the
same body location. Given matching modality and axis, similarity is the
SENSOR_PLACEMENT body-location label: 1.0 for an exact location string,
otherwise looked up in REGION_PROXIMITY (a small hand-curated anatomical
adjacency table — not inferred from data).

SENSOR_PLACEMENT entries are confirmed per-dataset from the exact code that
produces preprocessed_data/*.pkl:
  - HARTH/HAR70plus, capture24, WISDM/WISDM2, USC-HAD: unambiguous from each
    loader's own column list (see scripts/data_preprocess.py, data/*/​*_Loader.py).
  - Opportunity: derived from the OPPORTUNITY UCI dataset's official
    column_names.txt, traced through this repo's exact select_columns_opp
    deletions in data/Opportunity/Opportunity_Loader.py (verified
    position-by-position against the downloaded dataset, not reconstructed
    from memory).
  - Skoda: the dataset's own community documentation states the per-sensor
    column mapping was lost, so only the coarse fact "somewhere on the right
    arm" is registered, with axis=None. Since axis=None never satisfies the
    strict axis-match requirement above, 'drop' will correctly find no valid
    match for Skoda channels — use --channel_adapt pca for Skoda instead.
"""

import re
from typing import Dict, List, NamedTuple, Optional, Tuple


class Channel(NamedTuple):
    location: str             # body-region label, optionally side-prefixed
                              # (e.g. 'wrist', 'right_wrist', 'back', 'hip')
    modality: str             # physical quantity (e.g. 'accel', 'gyro', 'mag')
    axis: Optional[str]       # 'x' | 'y' | 'z' | None (None = unknown/scalar)


def _block(location: str, modality: str) -> List[Channel]:
    return [Channel(location, modality, axis) for axis in ('x', 'y', 'z')]


def _imu_block(location: str) -> List[Channel]:
    """accel + gyro + mag, 3 axes each — one OPPORTUNITY body-worn IMU."""
    return _block(location, 'accel') + _block(location, 'gyro') + _block(location, 'mag')


def _shoe_block(location: str) -> List[Channel]:
    """16 channels of one OPPORTUNITY L-SHOE/R-SHOE IMU (5 triaxial groups + compass)."""
    modalities = ('eu', 'nav_acc', 'body_acc', 'angvel_body', 'angvel_nav')
    channels = [ch for m in modalities for ch in _block(location, m)]
    channels.append(Channel(location, 'compass', None))
    return channels


SENSOR_PLACEMENT: Dict[str, List[Channel]] = {
    'HARTH':         _block('back', 'accel') + _block('thigh', 'accel'),
    'HAR70plus':     _block('back', 'accel') + _block('thigh', 'accel'),
    'capture24':     _block('wrist', 'accel'),
    'capture24mini': _block('wrist', 'accel'),
    'WISDM':         _block('pocket', 'accel'),
    'WISDM2':        _block('pocket', 'accel'),
    'USC_HAD':       _block('hip', 'accel') + _block('hip', 'gyro'),
    'Opportunity': (
        # 12 standalone accelerometers (raw columns 2-37), 3 axes each = 36 channels
        _block('right_knee', 'accel') + _block('hip', 'accel') +
        _block('left_upper_arm', 'accel') + _block('right_upper_arm', 'accel') +
        _block('left_hand', 'accel') + _block('back', 'accel') +
        _block('right_knee', 'accel') + _block('right_wrist', 'accel') +
        _block('right_upper_arm', 'accel') + _block('left_upper_arm', 'accel') +
        _block('left_wrist', 'accel') + _block('right_hand', 'accel') +
        # 5 body-worn IMUs (accel+gyro+mag, quaternions dropped), 9 channels each = 45
        _imu_block('back') + _imu_block('right_upper_arm') +
        _imu_block('right_forearm') + _imu_block('left_upper_arm') +
        _imu_block('left_forearm') +
        # 2 shoe IMUs, 16 channels each = 32
        _shoe_block('left_foot') + _shoe_block('right_foot')
    ),
    'Skoda': [Channel('right_arm', 'accel', None)] * 60,
}

assert len(SENSOR_PLACEMENT['Opportunity']) == 113, 'Opportunity registry must have 113 channels'
assert len(SENSOR_PLACEMENT['Skoda']) == 60, 'Skoda registry must have 60 channels'


# Hand-curated anatomical adjacency for region labels that actually appear
# above with a known axis (Skoda's 'right_arm' is excluded — axis=None
# already blocks it from ever matching, see module docstring).
REGION_PROXIMITY: Dict[frozenset, float] = {
    frozenset({'wrist', 'forearm'}):    0.8,
    frozenset({'wrist', 'hand'}):       0.6,
    frozenset({'wrist', 'upper_arm'}):  0.4,
    frozenset({'wrist', 'hip'}):        0.15,
    frozenset({'wrist', 'thigh'}):      0.1,
    frozenset({'wrist', 'pocket'}):     0.1,
    frozenset({'wrist', 'back'}):       0.1,
    frozenset({'wrist', 'knee'}):       0.1,
    frozenset({'wrist', 'foot'}):       0.05,
    frozenset({'forearm', 'upper_arm'}): 0.6,
    frozenset({'forearm', 'hand'}):     0.5,
    frozenset({'upper_arm', 'hand'}):   0.3,
    frozenset({'hip', 'thigh'}):        0.7,
    frozenset({'hip', 'pocket'}):       0.6,
    frozenset({'hip', 'back'}):         0.4,
    frozenset({'hip', 'knee'}):         0.3,
    frozenset({'thigh', 'pocket'}):     0.5,
    frozenset({'thigh', 'knee'}):       0.5,
    frozenset({'thigh', 'back'}):       0.2,
    frozenset({'back', 'knee'}):        0.1,
    frozenset({'foot', 'knee'}):        0.3,
    frozenset({'foot', 'thigh'}):       0.15,
}
_DEFAULT_REGION_SCORE = 0.05  # unlisted region pair: weak tiebreaker, not a real similarity claim
_LATERALITY_DISCOUNT  = 0.9   # same region, opposite recorded side (e.g. left_wrist vs right_wrist)

_SIDES = ('left', 'right')
_DATA_NAME_RE = re.compile(r'^_DA_(.+?)_\d+_\d+(?:-full)?$')


def _dataset_key(data_name: str) -> str:
    """'_DA_HAR70plus_256_00' -> 'HAR70plus'."""
    m = _DATA_NAME_RE.match(data_name)
    return m.group(1) if m else data_name


def get_placement(data_name: str) -> Optional[List[Channel]]:
    """Registered per-channel metadata for a dataset, or None if unregistered."""
    return SENSOR_PLACEMENT.get(_dataset_key(data_name))


def _split_side(location: str) -> Tuple[Optional[str], str]:
    for side in _SIDES:
        prefix = side + '_'
        if location.startswith(prefix):
            return side, location[len(prefix):]
    return None, location


def _location_score(loc_a: str, loc_b: str) -> float:
    if loc_a == loc_b:
        return 1.0
    side_a, region_a = _split_side(loc_a)
    side_b, region_b = _split_side(loc_b)
    score = 1.0 if region_a == region_b else \
        REGION_PROXIMITY.get(frozenset({region_a, region_b}), _DEFAULT_REGION_SCORE)
    if side_a is not None and side_b is not None and side_a != side_b:
        score *= _LATERALITY_DISCOUNT
    return score


def _channel_score(a: Channel, b: Channel) -> float:
    """0 unless modality and axis match exactly; otherwise the location similarity."""
    if a.axis is None or b.axis is None or a.axis != b.axis or a.modality != b.modality:
        return 0.0
    return _location_score(a.location, b.location)


def select_channels(finetune_data_name: str, finetune_num_channels: int,
                    pretrain_data_name: str, pretrain_num_channels: int) -> List[int]:
    """Pick pretrain_num_channels finetune channels best matching the pretrain
    channel set, by sensor-placement metadata only (see module docstring).

    Returns selected_indices where selected_indices[j] is the finetune raw-
    channel index assigned to pretrain channel j, so
    X_finetune[..., selected_indices] lines up positionally with the
    pretrained input projection's columns.

    Raises ValueError if either dataset has no registered placement, if the
    registered pretrain placement length doesn't match pretrain_num_channels
    (e.g. the pretrain checkpoint used --pca_components, so its input
    channels are PCA components rather than physical sensors and metadata
    matching is not meaningful), or if no valid (nonzero-score) match exists
    for one or more pretrain channels.
    """
    finetune_placement = get_placement(finetune_data_name)
    if finetune_placement is None:
        raise ValueError(
            f"--channel_adapt drop requires a registered sensor placement for "
            f"'{finetune_data_name}' in SENSOR_PLACEMENT (src/channel_adapt.py). "
            f"Use --channel_adapt pca instead, or add the placement.")
    if len(finetune_placement) != finetune_num_channels:
        raise ValueError(
            f"SENSOR_PLACEMENT['{_dataset_key(finetune_data_name)}'] has "
            f"{len(finetune_placement)} entries but the dataset has "
            f"{finetune_num_channels} raw channels — registry is out of date.")

    pretrain_placement = get_placement(pretrain_data_name)
    if pretrain_placement is None:
        raise ValueError(
            f"--channel_adapt drop requires a registered sensor placement for the "
            f"pretrain dataset '{pretrain_data_name}' in SENSOR_PLACEMENT "
            f"(src/channel_adapt.py). Use --channel_adapt pca instead, or add the placement.")
    if len(pretrain_placement) != pretrain_num_channels:
        raise ValueError(
            f"Pretrain checkpoint has {pretrain_num_channels} input channels but "
            f"'{pretrain_data_name}' is registered with {len(pretrain_placement)} placement "
            f"entries — most likely --pca_components was used at pretrain time, so its input "
            f"channels are PCA components, not physical sensors, and metadata matching is not "
            f"meaningful. Use --channel_adapt pca instead.")

    k = pretrain_num_channels
    D = finetune_num_channels
    if k >= D:
        return list(range(D))

    score = [[_channel_score(finetune_placement[fi], pretrain_placement[pj]) for pj in range(k)]
             for fi in range(D)]

    # Greedy stable matching: repeatedly take the best remaining (finetune, pretrain) pair.
    pairs = sorted(((score[fi][pj], fi, pj) for fi in range(D) for pj in range(k)),
                   key=lambda t: -t[0])
    selected: List[Optional[int]] = [None] * k
    used_finetune = set()
    for s, fi, pj in pairs:
        if s <= 0.0:
            break  # remaining pairs are all zero-score (modality/axis mismatch)
        if selected[pj] is not None or fi in used_finetune:
            continue
        selected[pj] = fi
        used_finetune.add(fi)
        if all(x is not None for x in selected):
            break

    unmatched = [pj for pj, fi in enumerate(selected) if fi is None]
    if unmatched:
        labels = [f'{pretrain_placement[pj].location}:{pretrain_placement[pj].modality}'
                 f'{pretrain_placement[pj].axis}' for pj in unmatched]
        raise ValueError(
            f"--channel_adapt drop found no finetune channel with matching modality+axis "
            f"for pretrain channel(s) {labels}. Use --channel_adapt pca instead, or extend "
            f"SENSOR_PLACEMENT / REGION_PROXIMITY in src/channel_adapt.py.")
    return selected


def assign_pretrain_columns(
        finetune_data_name: str, finetune_num_channels: int,
        pretrain_data_name: str, pretrain_num_channels: int,
) -> List[Optional[int]]:
    """For each finetune channel, find the best-matching pretrain column index.

    Used by the 'expand' channel_adapt strategy: the returned assignment drives
    initialisation of a wider input projection by copying pretrain weight columns.

    assignment[fi] = pj  — finetune channel fi copies pretrain column pj's weights.
    assignment[fi] = None — no valid (nonzero-score) match; that column keeps its
                           random initialisation from EncoderNView.__init__.

    Unlike select_channels (drop), multiple finetune channels can map to the same
    pretrain column — no 1-to-1 constraint. Both back_x and thigh_x can both copy
    the pretrained wrist_x weights, for example.

    Raises ValueError on missing/mismatched placement registries (same conditions
    as select_channels).
    """
    finetune_placement = get_placement(finetune_data_name)
    if finetune_placement is None:
        raise ValueError(
            f"--channel_adapt expand requires a registered sensor placement for "
            f"'{finetune_data_name}' in SENSOR_PLACEMENT (src/channel_adapt.py). "
            f"Use --channel_adapt pca instead, or add the placement.")
    if len(finetune_placement) != finetune_num_channels:
        raise ValueError(
            f"SENSOR_PLACEMENT['{_dataset_key(finetune_data_name)}'] has "
            f"{len(finetune_placement)} entries but the dataset has "
            f"{finetune_num_channels} raw channels — registry is out of date.")

    pretrain_placement = get_placement(pretrain_data_name)
    if pretrain_placement is None:
        raise ValueError(
            f"--channel_adapt expand requires a registered sensor placement for the "
            f"pretrain dataset '{pretrain_data_name}' in SENSOR_PLACEMENT "
            f"(src/channel_adapt.py). Use --channel_adapt pca instead, or add the placement.")
    if len(pretrain_placement) != pretrain_num_channels:
        raise ValueError(
            f"Pretrain checkpoint has {pretrain_num_channels} input channels but "
            f"'{pretrain_data_name}' is registered with {len(pretrain_placement)} placement "
            f"entries — most likely --pca_components was used at pretrain time, so its input "
            f"channels are PCA components, not physical sensors, and metadata matching is not "
            f"meaningful. Use --channel_adapt pca instead.")

    k = pretrain_num_channels
    D = finetune_num_channels
    assignment: List[Optional[int]] = []
    for fi in range(D):
        scores = [_channel_score(finetune_placement[fi], pretrain_placement[pj])
                  for pj in range(k)]
        best_pj = max(range(k), key=lambda pj: scores[pj])
        assignment.append(best_pj if scores[best_pj] > 0.0 else None)
    return assignment
