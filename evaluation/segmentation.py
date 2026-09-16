"""
U-Net Architecture Skeleton — Cloud Segmentation
==================================================
Project: AI-Based Cloud Detection and Localization in Satellite Images
Component: AI Model Development
Target dataset (not yet connected): 38-Cloud (Landsat 8, 384x384 patches, R/G/B/NIR)

STATUS: ARCHITECTURE SKELETON ONLY.
    - No dataset is loaded or downloaded here.
    - No training loop is included.
    - No accuracy/IoU/Dice numbers are reported (none exist yet — nothing has been trained).
    - This file only defines and prints the model structure so it is ready to plug
      a real tf.data pipeline into once the dataset is finalized.

Framework: TensorFlow / Keras (functional API)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# 1. CONFIG — placeholder values, update once the real data pipeline exists
# ---------------------------------------------------------------------------

# Placeholder input shape: (height, width, channels).
# 384x384 matches the 38-Cloud patch size. 4 channels = Red, Green, Blue, NIR
# (the four Landsat-8 bands the 38-Cloud dataset provides).
# NOTE: this is a placeholder based on the dataset's documented spec —
# it has not yet been verified against an actual loaded batch.
INPUT_SHAPE = (384, 384, 4)

# Number of filters in the first encoder block. Doubles at each downsampling
# stage (standard U-Net design): 64 -> 128 -> 256 -> 512 -> 1024 (bottleneck).
BASE_FILTERS = 64


# ---------------------------------------------------------------------------
# 2. BUILDING BLOCKS
# ---------------------------------------------------------------------------

def conv_block(x, filters, block_name):
    """
    Two 3x3 convolutions, each followed by BatchNorm + ReLU.
    This is the core repeated unit used in every encoder/decoder stage.
    """
    x = layers.Conv2D(filters, 3, padding="same", name=f"{block_name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn1")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu1")(x)

    x = layers.Conv2D(filters, 3, padding="same", name=f"{block_name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)
    x = layers.Activation("relu", name=f"{block_name}_relu2")(x)
    return x


def encoder_block(x, filters, block_name):
    """
    One downsampling stage of the U-Net encoder.

    Returns:
        skip       - the feature map BEFORE pooling (saved for the matching
                     decoder stage via a skip connection)
        pooled     - the feature map AFTER 2x2 max-pooling (passed deeper
                     into the network)
    """
    skip = conv_block(x, filters, block_name)
    pooled = layers.MaxPooling2D(pool_size=(2, 2), name=f"{block_name}_pool")(skip)
    return skip, pooled


def decoder_block(x, skip, filters, block_name):
    """
    One upsampling stage of the U-Net decoder.

    Steps:
        1. Upsample the incoming feature map (transposed convolution).
        2. Concatenate it with the matching encoder skip connection —
           this is what lets U-Net recover fine spatial detail (cloud
           edges/boundaries) that pooling would otherwise lose.
        3. Run it through a conv_block to blend the combined features.
    """
    x = layers.Conv2DTranspose(filters, 2, strides=(2, 2), padding="same",
                                name=f"{block_name}_upconv")(x)
    x = layers.Concatenate(name=f"{block_name}_concat")([x, skip])
    x = conv_block(x, filters, block_name)
    return x


# ---------------------------------------------------------------------------
# 3. FULL U-NET MODEL
# ---------------------------------------------------------------------------

def build_unet(input_shape=INPUT_SHAPE, base_filters=BASE_FILTERS):
    """
    Assembles the full U-Net: encoder -> bottleneck -> decoder -> output.

    Returns a compiled-but-untrained tf.keras.Model. No weights are trained
    here; no fit()/compile()-with-optimizer call is made in this file.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ---- Encoder (downsampling path) ----
    # Each stage halves the spatial resolution and doubles the filter count.
    skip1, p1 = encoder_block(inputs, base_filters * 1, "enc1")      # 384 -> 192
    skip2, p2 = encoder_block(p1,     base_filters * 2, "enc2")      # 192 -> 96
    skip3, p3 = encoder_block(p2,     base_filters * 4, "enc3")      # 96  -> 48
    skip4, p4 = encoder_block(p3,     base_filters * 8, "enc4")      # 48  -> 24

    # ---- Bottleneck ----
    # Deepest point of the network: smallest spatial resolution, richest
    # feature representation. No pooling after this — decoding starts next.
    bottleneck = conv_block(p4, base_filters * 16, "bottleneck")     # 24  -> 24

    # ---- Decoder (upsampling path) ----
    # Each stage doubles the spatial resolution back up and halves filters,
    # fusing in the matching encoder skip connection at every stage.
    d4 = decoder_block(bottleneck, skip4, base_filters * 8, "dec4")  # 24  -> 48
    d3 = decoder_block(d4,         skip3, base_filters * 4, "dec3")  # 48  -> 96
    d2 = decoder_block(d3,         skip2, base_filters * 2, "dec2")  # 96  -> 192
    d1 = decoder_block(d2,         skip1, base_filters * 1, "dec1")  # 192 -> 384

    # ---- Output layer ----
    # 1x1 convolution collapses features down to a single channel: one
    # cloud-probability value per pixel. Sigmoid squashes it to [0, 1] for
    # binary segmentation (cloud vs. not-cloud). Thresholding (e.g. at 0.5)
    # happens LATER, at inference time — not inside this model definition.
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="cloud_mask_output")(d1)

    model = Model(inputs=inputs, outputs=outputs, name="unet_cloud_segmentation")
    return model


# ---------------------------------------------------------------------------
# 4. SKELETON CHECK — build the model and print its structure only
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_unet()
    model.summary()

    # No model.compile(), no model.fit(), no dataset loading here.
    # This block only confirms the architecture builds and shows its shape
    # at every layer — useful for sanity-checking before data is connected.
    print("\nModel built successfully. Input shape:", INPUT_SHAPE)
    print("Output shape:", model.output_shape, "(per-pixel cloud probability map)")