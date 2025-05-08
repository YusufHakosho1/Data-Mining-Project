#check if the environment libs working or not 
import sys

# 1. Show which interpreter is running
print("Interpreter:", sys.executable)

# 2. pandas version
import pandas as pd
print("pandas:", pd.__version__)

# 3. matplotlib version (import top-level module, not pyplot)
import matplotlib
print("matplotlib:", matplotlib.__version__)  # __version__ lives here, not in pyplot :contentReference[oaicite:1]{index=1}

# 4. scikit-learn version
import sklearn
print("scikit-learn:", sklearn.__version__)

# 5. SciPy version
import scipy
print("SciPy:", scipy.__version__)

# 6. mlxtend version
import mlxtend
print("mlxtend:", mlxtend.__version__)

# 7. NetworkX version
import networkx as nx
print("NetworkX:", nx.__version__)
