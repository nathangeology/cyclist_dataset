import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool

from scipy import ceil, sqrt

def clusters2classes(cards_g1, clustLabels_g1, clust_name_g1, cards_g2,
                     clustLabels_g2, clust_name_g2,  clusters2class_dict):
    labeled_card = {}

    for cardclass in clusters2class_dict['fromOutliers']:
        labeled_card[cardclass] = {}

        for col in list(cards_g1.keys()):
            labeled_card[cardclass].update({col: []})

        for clstr in clusters2class_dict['fromOutliers'][cardclass]:
            inds = np.where(clustLabels_g1 == clstr)[0]

            clust_name_g1[clstr] = clust_name_g1[clstr] + ', ' + cardclass

            for col in list(cards_g1.keys()):
                if len(labeled_card[cardclass][col]) == 0:
                    labeled_card[cardclass][col] = cards_g1[col][inds, :]
                else:
                    labeled_card[cardclass][col] = np.append(labeled_card[cardclass][col],
                                                             cards_g1[col][inds, :], axis=0)


    for cardclass in clusters2class_dict['fromNormals']:
        for clstr in clusters2class_dict['fromNormals'][cardclass]:
            inds = np.where(clustLabels_g2 == clstr)[0]

            clust_name_g2[clstr] = clust_name_g2[clstr] + ', ' + cardclass

            for col in list(cards_g2.keys()):
                if len(labeled_card[cardclass][col]) == 0:
                    labeled_card[cardclass][col] = cards_g2[col][inds, :]
                else:
                    labeled_card[cardclass][col] = np.append(labeled_card[cardclass][col],
                                                             cards_g2[col][inds, :],  axis=0)

    return  labeled_card, clust_name_g1, clust_name_g2


def plot_clustered_cards(card_data,col_x, col_y, data_labels, clust_name):
    num_clust = len(clust_name)
    ncolp = np.min([3, ceil(sqrt(num_clust))])
    nrowp = np.ceil(num_clust / ncolp)
    fig2 = plt.figure(figsize=(min([ncolp*6,30]), nrowp*5))

    load_labels = np.zeros((data_labels.shape[0], 1))
    inds_data = {}
    for clstr in range(num_clust):
        inds_clstr = np.where(data_labels == clstr)
        inds_clstr = inds_clstr[0]

        load_labels[inds_clstr] = clstr

        inds_data.update({clstr: inds_clstr})

        x = card_data[col_x][inds_clstr, :]
        y = card_data[col_y][inds_clstr, :]

        ax1 = plt.subplot(nrowp, ncolp, clstr + 1)
        for j in range(x.shape[0]):
            ax1.plot(x[j, :], y[j, :])
            ax1.tick_params(colors='w', direction='out')

        ax1.set_title(clust_name[clstr], color='w')
    plt.show()


def perim(x, y):
    # Calculates perimeter of a closed polygon
    dx2 = (np.append(x[1:], x[0]) - x) ** 2
    dy2 = (np.append(y[1:], y[0]) - y) ** 2
    return np.sum((dx2 + dy2) ** 0.5)

def periphery_area(x, y):
    # Calculates peripheral area above and below the card
    ind1 = np.where(x == x.max())
    m = ind1[0][0]

    area_above = x.max() * y.max() - 0.5 * np.sum((x[1:m] - x[:m - 1]) * (y[:m - 1] + y[1:m]))
    area_below = 0.5 * np.sum((x[m:-1] - x[m + 1:]) * (y[m:-1] + y[m + 1:]))

    return area_above, area_below

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def enumerate_cards(card_data):
    POCdh_all_y = []
    vec_size = []
    NROW, NCOl = card_data.shape

    surfLoadMax = np.empty((0, 1))
    surfLoadMin = np.empty((0, 1))

    minload = np.empty((0, 1))
    load_atMaxPos = np.empty((0, 1))
    maxload = np.empty((0, 1))
    cumload = np.empty((0, 1))

    maxpos_atMinLoad = np.empty((0, 1))
    minpos_atMaxLoad = np.empty((0, 1))
    maxpos = np.empty((0, 1))
    minpos = np.empty((0, 1))

    ctr_load = np.empty((0, 1))
    ctr_pos = np.empty((0, 1))
    perimeter = np.empty((0, 1))
    area_above = np.empty((0, 1))
    area_below = np.empty((0, 1))

    NT = 200  # len(dhCard[0])
    load_all = np.empty((0, NT))
    pos_all = np.empty((0, NT))

    load_nall = np.empty((0, NT))
    pos_nall = np.empty((0, NT))


    for t in range(NROW):
        try:
            dhCard = eval(card_data['DownholeCardB'].iloc[t])
            surfCard = eval(card_data['SurfaceCardB'].iloc[t])
        except:
            continue

        if len(dhCard[0]) != NT:
            continue

        # load = list(reversed([float(i) for i in dhCard[0]]))
        # pos = list(reversed([float(i) for i in dhCard[1]]))

        load = np.asarray([float(i) for i in dhCard[0]])
        # load = load - load.min()

        pos = np.asarray([float(i) for i in dhCard[1]])
        # pos = pos - pos.min()

        #        load = [float(i) for i in POCdhCard[0]]
        #        pos = [float(i) for i in POCdhCard[1]]
        surf_load = np.asarray([float(i) for i in surfCard[0]])
        surfLoadMax = np.append(surfLoadMax, surf_load.max())
        surfLoadMin = np.append(surfLoadMin, surf_load.min())

        vec_size.append(len(pos))
        #         load_n = np.array( [(i - min(load))/(max(load)-min(load)) for i in load] )
        #         pos_n =  np.array( [(i - min(pos)) / (max(pos) - min(pos)) for i in pos] )
        load_n = (load - load.min()) / (load.max()- load.min())
        pos_n = (pos - pos.min())    / (pos.max() - pos.min())

        ctr_load = np.append(ctr_load, load_n.mean())
        ctr_pos = np.append(ctr_pos, pos_n.mean())
        perimeter = np.append( perimeter, perim(pos - pos.min(), load - load.min()) )
        aa, ab = periphery_area(pos - pos.min(), load - load.min())
        area_above = np.append(area_above, aa.mean())
        area_below = np.append(area_below, ab.mean())

        minload = np.append(minload, load.min())
        maxload = np.append(maxload, load.max())

        cumload = np.append(cumload, load.sum())

        ind0 = np.where(load_n <= 0.05)
        pos_load0 = pos[ind0[0]]
        maxpos_atMinLoad = np.append(maxpos_atMinLoad, [max(pos_load0)])

        ind0 = np.where(load_n >= 0.95)
        pos_load1 = pos[ind0[0]]
        minpos_atMaxLoad = np.append(minpos_atMaxLoad, [min(pos_load1)])

        ind0 = np.where(pos_n >= 0.95)
        load_pos0 = load[ind0[0]]
        load_atMaxPos = np.append(load_atMaxPos, [min(load_pos0)])



        maxpos = np.append(maxpos, pos.max())
        minpos = np.append(minpos, pos.min())

        load_all = np.append(load_all, [load], axis=0)
        pos_all = np.append(pos_all, [pos], axis=0)

        load_nall = np.append(load_nall, [load_n], axis=0)
        pos_nall = np.append(pos_nall, [pos_n], axis=0)

    output = {'load': load_all,
              'load_norm': load_nall,
              'position': pos_all,
              'position_norm': pos_nall,
              'load_atMaxPos': load_atMaxPos[:, np.newaxis],
              'surfLoadMax': surfLoadMax[:, np.newaxis],
              'surfLoadMin': surfLoadMin[:, np.newaxis],
              'position_min': minpos[:, np.newaxis],
              'position_max': maxpos[:, np.newaxis],
              'maxPosition_at_minLoad': maxpos_atMinLoad[:, np.newaxis],
              'minPosition_at_maxLoad': minpos_atMaxLoad[:, np.newaxis],
              'load_min': minload[:, np.newaxis],
              'load_max': maxload[:, np.newaxis],
              'load_sum': cumload[:, np.newaxis],
              'load_norm_center': ctr_load[:, np.newaxis],
              'position_norm_center': ctr_pos[:, np.newaxis],
              'perimeter': perimeter[:, np.newaxis],
              'area_above': area_above[:, np.newaxis],
              'area_below': area_below[:, np.newaxis],
              'cumsum_load_norm': np.cumsum(load_nall, axis=1)[:, 0::10],
              'cumsum_position_norm': np.cumsum(pos_nall, axis=1)[:, 0::10]
              }

    area = [PolyArea(pos_all[i, :], load_all[i, :]) for i in range(load_all.shape[0])]
    output['area'] = np.asarray(area)[:, np.newaxis]

    return output
