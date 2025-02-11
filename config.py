# =========================
# Configuration Settings
# =========================

import os
import warnings

# Suppress TensorFlow logs (set "2" to suppress INFO & WARNING, "3" to suppress all logs)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Improve performance by enabling oneDNN optimizations (only if using Intel CPUs)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

# Set NumPy print options for better readability
import numpy as np
np.set_printoptions(suppress=True, precision=4)

# Pandas display settings for better DataFrame visibility
import pandas as pd
pd.set_option("display.float_format", "{:.4f}".format)  # Format float values

# Seaborn visualization settings
import seaborn as sns
sns.set_style("whitegrid")  # Set a clean grid style
sns.set_context("notebook")  # Set context for better font sizes in plots

# Matplotlib settings to improve plotting
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6)  # Default figure size
plt.rcParams["axes.grid"] = True  # Enable grid for better readability
