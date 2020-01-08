import numpy as np
import os
import pandas as pd
import helperFuncs as funcH

#np.set_printoptions(precision=2)
def printIndicesPerCluster(cluster_runs):
    clustRunCount, sampleCount = cluster_runs.shape
    print("there are {:d} clusters for {:d} samples\n".format(clustRunCount, sampleCount))
    for cID in range(0,clustRunCount):
        clustCur = cluster_runs[cID,:].squeeze()
        print("clustID({:d}): {}".format(cID+1, clustCur))
        klustIDs = np.unique(clustCur).squeeze()
        print("*-*-*-*")
        for k in klustIDs:
            indList = funcH.getInds(clustCur, k);
            print("clustGroup({:d}), clustID({:d}): {}".format(cID+1, k, indList+1))

# F_05
def getClusterVariables(clustVec, verbose=0):
    clustIDs = np.unique(clustVec).squeeze()
    uniqClusterCount = len(clustIDs)
    if verbose > 7:
        print(
            "+++++getClusterVariables - input:clustVec({}), out:uniqClusterCount({:d}), clustIDs({}) ".format(clustVec,
                                                                                                              uniqClusterCount,
                                                                                                              clustIDs))
    return clustIDs, uniqClusterCount

def getClusterVariablesFromAListOfClusters(cluster_runs, clusterID, verbose=0):
    clustVec = cluster_runs[clusterID, :].squeeze()
    clustIDs, uniqClusterCount = getClusterVariables(clustVec)
    return clustVec, clustIDs, uniqClusterCount

# F_04
def calc_uncert_samples_ci_cj(clust_indices_i, clust_indices_j, logType=0, verbose=0):
    i_j = set(clust_indices_i).intersection(set(clust_indices_j))
    p_ci_cj = len(i_j) / len(clust_indices_i)
    if verbose > 6:
        print("++++calc_uncert_samples_ci_cj - intersect set({}), p_ci_cj({:.4f})".format(i_j, p_ci_cj))
    #retVal = np.log(p_ci_cj) if logType == 0 else np.log2(p_ci_cj)
    return 0 if len(i_j) == 0 else -p_ci_cj * (np.log(p_ci_cj) if logType == 0 else np.log2(p_ci_cj))

# F_03
def calc_uncert_cluster_i_j(klust_i, klust_j, logType=0, verbose=0):
    clustIDs_i, uniqClusterCount_i = getClusterVariables(klust_i)
    clustIDs_j, uniqClusterCount_j = getClusterVariables(klust_j)
    H_i = np.zeros([uniqClusterCount_i, uniqClusterCount_j], dtype=float)
    for ii in range(0, uniqClusterCount_i):
        clust_indices_i = funcH.getInds(klust_i, clustIDs_i[ii]);
        for ji in range(0, uniqClusterCount_j):
            clust_indices_j = funcH.getInds(klust_j, clustIDs_j[ji]);
            ij_un = calc_uncert_samples_ci_cj(clust_indices_i, clust_indices_j, logType=logType, verbose=verbose)
            if verbose > 5:
                print("+++calc_uncert_cluster_i_j - input:klust_i({}),klust_j({})".format(klust_i, klust_j))
                print(
                    "                             var:clust_indices_i({})-clustIDs_i[ii]({}), clust_indices_j({})-clustIDs_j[ji]({}) ".format(
                        clust_indices_i, clustIDs_i[ii], clust_indices_j, clustIDs_j[ji]))
                print("                             out:ij_un({:.4f})".format(ij_un))
            H_i[ii, ji] = ij_un
    return H_i.squeeze()

# F_02
def calc_uncert_cluster_i(klust_id_i, cluster_runs, logType=0, verbose=0):
    clustRunCount, sampleCount = cluster_runs.shape
    clustVec_i, clustIDs_i, uniqClusterCount_i = getClusterVariablesFromAListOfClusters(cluster_runs, klust_id_i)
    Hi = []
    for j in range(0, clustRunCount):
        if j != klust_id_i:
            clustVec_j = cluster_runs[j, :].squeeze()
            H_ij = calc_uncert_cluster_i_j(clustVec_i, clustVec_j, logType=logType, verbose=verbose)
            if verbose > 4:
                print(
                    "++calc_uncert_cluster_i  - input:klust_id_i({:d}), var:j({:d}), out:H_ij({}), ".format(klust_id_i,
                                                                                                            j, H_ij))
        else:
            if verbose > 4:
                print(
                    "++calc_uncert_cluster_i  - input:klust_id_i({:d}), var:j({:d}), out:zeros, ".format(klust_id_i, j))
            H_ij = np.zeros([uniqClusterCount_i, uniqClusterCount_i], dtype=float)
        if verbose > 4:
            print("++append H_ij({}) into Hi({})".format(np.shape(H_ij), np.shape(Hi)))
        Hi = H_ij if len(Hi) == 0 else np.asarray(np.hstack((Hi, H_ij)))
        if verbose > 4:
            print("++H_i new size({})".format(np.shape(Hi)))

    return Hi.squeeze(), uniqClusterCount_i

# F_01
def calc_uncert_cluster_set(cluster_runs, logType=0, verbose=0):
    clustRunCount, sampleCount = cluster_runs.shape
    H = []
    clustCounts = []
    for i in range(0, clustRunCount):
        Hi, uniqClusterCount_i = calc_uncert_cluster_i(i, cluster_runs, logType=logType, verbose=verbose)
        if verbose > 3:
            print("+H_i with shape ({})".format(np.shape(Hi)))
            print(Hi)
            print("+append Hi({}) into H({})".format(np.shape(Hi), np.shape(H)))
        H = Hi if len(H) == 0 else np.asarray(np.vstack((H, Hi)), dtype=float)
        if verbose > 3:
            print("+H new size({})".format(np.shape(H)))
        clustCounts.append(uniqClusterCount_i)
    if verbose > 3:
        print("+Final H with shape ({})".format(np.shape(H)))
        print(H)
        print("clustCounts({})".format(clustCounts))
    return np.sum(H, axis=1), clustCounts

def calc_display_uncert_cluster_set(cluster_runs, logType=0, verbose=0):
    clustUncertainity, clusterCounts = calc_uncert_cluster_set(cluster_runs, logType=logType, verbose=verbose)
    clustRunCount, sampleCount = cluster_runs.shape
    if verbose > 2:
        print("there are {:d} clusters for {:d} samples\n".format(clustRunCount, sampleCount))
    fr = 0
    for cID in range(0, clustRunCount):
        nc = clusterCounts[cID]
        to = fr + nc
        if verbose > 2:
            print("cluster({:d}), cl_un_wrt_ens({})".format(cID, clustUncertainity[fr:to]))
        fr = to

def calc_cluster_average_entropy(cluster_runs, logType=0, verbose=0):
    clustUncertainity, clusterCounts = calc_uncert_cluster_set(cluster_runs, logType=logType, verbose=verbose)
    clustRunCount, sampleCount = cluster_runs.shape

    cae_vec = np.zeros((clustRunCount,), dtype=float)

    if verbose > 1:
        print("there are {:d} clusters for {:d} samples\n".format(clustRunCount, sampleCount))
    fr = 0
    for cID in range(0, clustRunCount):
        nc = clusterCounts[cID]
        to = fr + nc

        cae_vec[cID] = np.sum(clustUncertainity[fr:to]).squeeze() / nc

        if verbose > 1:
            print("cluster({:d}), cl_un_wrt_ens({}), cluster_average_entropy({})".format(cID, clustUncertainity[fr:to],
                                                                                         cae_vec[cID]))
        fr = to
    return cae_vec

def calc_ensemble_driven_cluster_index(cluster_runs, tetaVal=0.2, verbose=0):
    clustUncertainity, clusterCounts = calc_uncert_cluster_set(cluster_runs, logType=2, verbose=verbose)
    clustRunCount, sampleCount = cluster_runs.shape

    eci_vec = []

    if verbose > 1:
        print("there are {:d} clusters for {:d} samples\n".format(clustRunCount, sampleCount))
    fr = 0
    for cID in range(0, clustRunCount):
        nc = clusterCounts[cID]
        to = fr + nc
        eci_vec_cid = np.zeros((nc,), dtype=float)
        if verbose > 1:
            print("[{}->{}]cluster({:d}), cl_un_wrt_ens({}), clusterCounts[{:d}]({})".format(fr, to, cID,
                                                                                             clustUncertainity[fr:to],
                                                                                             cID, clusterCounts[cID]))

        eci_vec_cid = np.exp(-clustUncertainity[fr:to] / (tetaVal * clusterCounts[cID]))
        eci_vec.append(eci_vec_cid)

        if verbose > 1:
            print("cluster({:d}), cl_un_wrt_ens({}), ensemble_driven_cluster_index({})".format(cID,
                                                                                               clustUncertainity[fr:to],
                                                                                               eci_vec_cid))
        fr = to
    return eci_vec, clusterCounts

def calc_quality_weight_basic_clustering(cluster_runs, logType=0, verbose=0):
    cae_vec = calc_cluster_average_entropy(cluster_runs, logType=logType, verbose=verbose)
    return np.exp(-cae_vec)

def createSamples(paperID, verbose=0):
    if paperID == 1:
        # cl_00 = np.array([1,2,3,4,5,6,7,8,9,0])
        cl_01 = np.array([1, 2, 1, 3, 2, 2, 3, 3, 1, 2])
        cl_02 = np.array([1, 3, 2, 1, 3, 4, 1, 2, 3, 4])
        cl_03 = np.array([1, 2, 1, 3, 4, 3, 1, 2, 4, 2])
        cluster_runs = cl_01
        cluster_runs = np.asarray(np.vstack((cluster_runs, cl_02)), dtype=int)
        cluster_runs = np.asarray(np.vstack((cluster_runs, cl_03)), dtype=int)
    elif paperID == 2:
        #cl_00 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        cl_01_2 = np.array([1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 3, 1, 3, 3, 3, 3])
        cl_02_2 = np.array([1, 1, 1, 1, 2, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3])
        cl_03_2 = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3])
        cluster_runs = cl_01_2
        cluster_runs = np.asarray(np.vstack((cluster_runs, cl_02_2)), dtype=int)
        cluster_runs = np.asarray(np.vstack((cluster_runs, cl_03_2)), dtype=int)
    if verbose > 0:
        printIndicesPerCluster(cluster_runs)
    return cluster_runs

def create_CA_matrix(cluster_runs, verbose=0):
    clustRunCount, sampleCount = cluster_runs.shape
    print("there are {:d} clusters for {:d} samples\n".format(clustRunCount, sampleCount))
    ca_mat = np.zeros((sampleCount, sampleCount), dtype=int)
    for ci in range(0, clustRunCount):
        # check if si&sj falls into same cluster
        clustCur, clustIDs, uniqClusterCount = getClusterVariablesFromAListOfClusters(cluster_runs, ci, verbose=verbose)
        for ki in range(0, uniqClusterCount):
            clusterIDCur = clustIDs[ki]
            sIDs = funcH.getInds(clustCur, clusterIDCur)
            sampleCount = len(sIDs)
            for si in range(0, sampleCount):
                for sj in range(si + 1, sampleCount):
                    # print("si({})-sj({})".format(si,sj))
                    id_i = sIDs[si]
                    id_j = sIDs[sj]
                    ca_mat[id_i, id_j] = ca_mat[id_i, id_j] + 1
    print(pd.DataFrame(ca_mat))

def create_LWCA_matrix(cluster_runs, eci_vec=None, verbose=0):
    if eci_vec is None:
        eci_vec, clusterCounts = calc_ensemble_driven_cluster_index(cluster_runs, tetaVal=0.5, verbose=verbose)
        print("eci_vec = ", eci_vec)
    clustRunCount, sampleCount = cluster_runs.shape
    if verbose > 0:
        print("there are {:d} clusters for {:d} samples\n".format(clustRunCount, sampleCount))
    lwca_mat = np.zeros((sampleCount, sampleCount), dtype=float)
    for ci in range(0, clustRunCount):
        # check if si&sj falls into same cluster
        if verbose > 0:
            print("eci_vec_cid = ", eci_vec[ci])
        clustCur, clustIDs, uniqClusterCount = getClusterVariablesFromAListOfClusters(cluster_runs, ci, verbose=verbose)
        for ki in range(0, uniqClusterCount):
            clusterIDCur = clustIDs[ki]
            sIDs = funcH.getInds(clustCur, clusterIDCur)
            sampleCount = len(sIDs)
            for si in range(0, sampleCount):
                for sj in range(si + 1, sampleCount):
                    # print("si({})-sj({})".format(si,sj))
                    id_i = sIDs[si]
                    id_j = sIDs[sj]
                    lwca_mat[id_i, id_j] = lwca_mat[id_i, id_j] + eci_vec[ci][ci]
    #print(pd.DataFrame(lwca_mat))
    return lwca_mat

# quality_weights = calc_quality_weight_basic_clustering(cluster_runs,logType=0,verbose=3)
# print("quality_weights = {}".format(quality_weights))
# calc_cluster_average_entropy(cluster_runs_2,logType=2,verbose=2)
# calc_ensemble_driven_cluster_index(cluster_runs_2,tetaVal=0.5,verbose=2)
# uc, clustCounts_2 = calc_uncert_cluster_set(cluster_runs_2,logType=2,verbose=3)
# print(uc)

#funcH.setPandasDisplayOpts()
#pd.options.display.float_format = '{:,.2f}'.format
#cluster_runs = createSamples(paperID=2, verbose=1)
#quality_weights, clusterCounts = calc_ensemble_driven_cluster_index(cluster_runs, tetaVal=0.5, verbose=1)
#create_CA_matrix(cluster_runs, verbose=0)
#create_LWCA_matrix(cluster_runs, eci_vec=None, verbose=0)