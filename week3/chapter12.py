import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from ISLP.cluster import compute_linkage

USArrests = get_rdataset('USArrests').data
USArrests
