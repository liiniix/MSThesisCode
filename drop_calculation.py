import matplotlib.pyplot as plt
import json
import numpy as np

with open(f'cora__epoch_200__0to30__PhD.json', 'r') as file:
    data = json.load(file)

    GCN = np.array(data['GCN'])
    HGAT = np.array(data['HGAT'])

    GCN_mean_first_15_layers = GCN[:17].mean()
    GCN_mean_last = GCN[17:].mean()

    HGAT_mean_first_15_layers = HGAT[:17].mean()
    HGAT_mean_last = HGAT[17:].mean()


    print((GCN_mean_last-GCN_mean_first_15_layers)/GCN_mean_first_15_layers)

    print((HGAT_mean_last-HGAT_mean_first_15_layers)/HGAT_mean_first_15_layers)