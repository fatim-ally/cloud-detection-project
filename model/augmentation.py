"""
Synchronized flip/rotation augmentation for the cloud-segmentation tf.data
pipeline (image + mask pairs). Add this alongside the existing pipeline code
-- it does not require changing how patches are loaded, only how the
training dataset is mapped before batching.

Not executed as part of training here -- Fatima will wire this into the
real pipeline and test it in Colab.
"""

import tensorflow as tf


def augment_image_mask(image, mask, seed):
    """
    Applies a random horizontal flip, random vertical flip, and random
    0/90/180/270-degree rotation to one (image, mask) pair, with the SAME
    random decisions applied to both -- this is what keeps every pixel in
    the mask aligned with the same pixel in the image after augmentation.

    image : tf.Tensor, shape (H, W, 4)  -- stacked Red/Green/Blue/NIR bands
    mask  : tf.Tensor, shape (H, W, 1)  -- binary cloud mask
    seed  : tf.Tensor, shape (2,), int32 or int64 -- one seed per example,
            see augment_dataset() below for how this is generated.

    Returns (image, mask), both transformed identically.
    """
    # Split the one incoming seed into three independent sub-seeds, so the
    # h-flip, v-flip, and rotation decisions don't all move together --
    # without this, a single seed reused for all three stateless_* calls
    # below would make the three "random" choices perfectly correlated.
    seeds = tf.random.split(seed, num=3)
    seed_hflip, seed_vflip, seed_rot = seeds[0], seeds[1], seeds[2]

    # --- Random horizontal flip (left-right), same decision for both ---
    image = tf.image.stateless_random_flip_left_right(image, seed=seed_hflip)
    mask = tf.image.stateless_random_flip_left_right(mask, seed=seed_hflip)

    # --- Random vertical flip (up-down), same decision for both ---
    image = tf.image.stateless_random_flip_up_down(image, seed=seed_vflip)
    mask = tf.image.stateless_random_flip_up_down(mask, seed=seed_vflip)

    # --- Random rotation restricted to 0/90/180/270 degrees ---
    # Deliberately NOT an arbitrary angle: on a square 384x384 patch, a
    # multiple-of-90-degree rotation needs no interpolation and no border
    # padding, so every augmented pixel is a real original pixel just moved
    # to a new position -- nothing is invented or blurred at the edges.
    # An arbitrary-angle rotation would require interpolating the binary
    # mask, turning crisp 0/255 edges into blurry in-between values.
    k = tf.random.stateless_uniform(shape=[], seed=seed_rot, minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    mask = tf.image.rot90(mask, k=k)

    return image, mask


def augment_dataset(dataset, seed_offset=0):
    """
    Wraps an existing (image, mask) tf.data.Dataset with synchronized
    flip/rotation augmentation.

    Apply this to the TRAINING dataset only -- never to validation or test,
    since those must reflect real, unaugmented performance.

    dataset     : tf.data.Dataset yielding (image, mask) pairs, unbatched
                  (call this before .batch(...)).
    seed_offset : int. Change this between experiments if you want a
                  different (but still reproducible) augmentation sequence;
                  keep it fixed across epochs/runs you want to compare.

    Returns a new tf.data.Dataset of the same (image, mask) structure.
    """
    # .enumerate() gives each example a unique, deterministic index, which
    # we turn into a per-example seed. This is what makes the stateless_*
    # random ops above both reproducible AND safe to use with
    # num_parallel_calls (ordinary tf.random ops don't mix well with
    # parallel/prefetched mapping -- stateless ops are the current
    # recommended pattern for tf.data augmentation).
    dataset = dataset.enumerate()

    def _map_fn(idx, example):
        image, mask = example
        seed = tf.stack([idx + seed_offset, 0])
        seed = tf.cast(seed, tf.int32)
        return augment_image_mask(image, mask, seed)

    return dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)


# -----------------------------------------------------------------------
# Example wiring into an existing pipeline (adjust names to match yours):
#
#   train_ds = augment_dataset(train_ds)          # BEFORE batching
#   train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#
#   val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # no augmentation
# -----------------------------------------------------------------------
