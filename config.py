# config.py
SEED         = 42
NUM_POINTS   = 1024
NUM_CLASSES  = 10        # confirm with your friend
IN_CHANNELS  = 6         # 3 for XYZ only, 6 for XYZ+RGB — confirm with friend
BATCH_SIZE   = 32
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
VFL_EMBED_DIM = 256      # agreed upon in the project spec
K_NEIGHBORS  = 20        # for DGCNN