import numpy as np
from hmmlearn import hmm

def create_hmm_trans_mat(n_components, transStepAllow, transitionModel='lr', verbose=0):
    if transitionModel == 'random':
        transmat_ = np.random.rand(n_components, n_components)
        row_sums = transmat_.sum(axis=1)
        transmat_ = transmat_ / row_sums[:, np.newaxis]
    if transitionModel == 'lr':
        transmat_ = np.zeros([n_components, n_components], dtype=float)
        for i in range(0, n_components):
            uptoCol = np.minimum(n_components, i + transStepAllow)
            stepCnt = uptoCol - i
            transVal = 1.0 / stepCnt
            if verbose > 1:
                print(i, uptoCol, stepCnt, transVal)
            for j in range(i, uptoCol):
                transmat_[i][j] = transVal
    if transitionModel == 'circular':
        transmat_ = np.zeros([n_components, n_components], dtype=float)
        transVal = 1.0 / transStepAllow
        for i in range(0, n_components):
            uptoCol = np.minimum(n_components, i + transStepAllow)
            stepCnt = uptoCol - i
            if verbose > 1:
                print(i, uptoCol, stepCnt, transVal)
            for j in range(i, uptoCol):
                transmat_[i][j] = transVal
            if uptoCol == n_components:
                if verbose > 1:
                    print("*", i, transStepAllow - stepCnt)
                for j in range(0, transStepAllow - stepCnt):
                    if verbose > 2:
                        print("**", i, j)
                    transmat_[i][j] = transVal

    return transmat_

def create_hmm_start_prob(n_components, startModel='firstOnly', verbose=0):
    if startModel == 'firstOnly':
        startprob_ = np.zeros([1, n_components], dtype=float).squeeze()
        startprob_[0] = 1.0
    elif startModel == 'decreasing':
        startprob_ = np.arange(1, n_components + 1).squeeze()
        startprob_ = -np.sort(-startprob_ / np.sum(startprob_))
    return startprob_

def createHMMModel(n_components, transStepAllow, n_iter=100, startModel='firstOnly', transitionModel='lr', verbose=0):
    # startprob_ , transmat_ = createHMMModel(n_components, transStepAllow, n_iter=100, startModel='firstOnly', transitionModel='lr')
    startprob_ = create_hmm_start_prob(n_components, startModel=startModel, verbose=verbose)
    if verbose > 0:
        print("hmm start probability vector [startModel={}]".format(startModel))
        print(startprob_)

    transmat_ = create_hmm_trans_mat(n_components, transStepAllow, transitionModel=transitionModel, verbose=verbose)
    if verbose > 0:
        print("hmm initial transition matrix [transitionModel={}, transStepAllow={:d}]".format(transitionModel,
                                                                                               transStepAllow))
        print(transmat_)

    _hmm_model_ = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter,
                                  covariance_type="diag", init_params="cm", params="cmt")

    startprob_ = create_hmm_start_prob(n_components=n_components, startModel=startModel, verbose=verbose)
    transmat_ = create_hmm_trans_mat(n_components=n_components, transStepAllow=transStepAllow,
                                     transitionModel=transitionModel, verbose=verbose)

    _hmm_model_.startprob_ = startprob_
    _hmm_model_.transmat_ = transmat_

    return _hmm_model_

def try_script01():
    n_components = 5
    transStepAllow = 2
    startModel = 'decreasing'  # 'firstOnly' 'decreasing'
    transitionModel = 'lr'  # 'random' 'lr' 'circular'
    verbose = 1
    _hmm_model_ = createHMMModel(n_components=n_components, transStepAllow=transStepAllow,
                                 startModel=startModel, transitionModel=transitionModel, verbose=verbose)

def try_script02_createSamplesFromModel():
    np.random.seed(42)
    model = hmm.GaussianHMM(n_components=3, covariance_type="full")
    model.startprob_ = np.array([0.6, 0.3, 0.1])
    model.transmat_ = np.array([[0.7, 0.2, 0.1],
                                [0.3, 0.5, 0.2],
                                [0.3, 0.3, 0.4]])
    model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
    model.covars_ = np.tile(np.identity(2), (3, 1, 1))
    X, Z = model.sample(100)
    print("X")
    print(X.shape)
    print("Z")
    print(Z.shape)

def try_script_03():
    X1 = [[0.5], [1.0], [-1.0], [0.42], [0.24]]
    X2 = [[2.4], [4.2], [0.5], [-0.24]]
    X = np.concatenate([X1, X2])
    lengths = [len(X1), len(X2)]
    model3 = hmm.GaussianHMM(n_components=3).fit(X, lengths)

def initialize_model_lr(n_components, n_iter=100):
    lr = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter,
                         covariance_type="diag", init_params="cm", params="cmt")
    #  lr.startprob_ = np.array([1.0, 0.0, 0.0])
    #  lr.transmat_ = np.array([[0.5, 0.5, 0.0],
    #                        [0.0, 0.5, 0.5],
    #                         [0.0, 0.0, 1.0]])
    startprob_ = create_hmm_start_prob(n_components, startModel='firstOnly', verbose=1)
    transmat_ = create_hmm_trans_mat(n_components, transStepAllow=2, transitionModel='lr', verbose=1)

    lr.startprob_ = startprob_
    lr.transmat_ = transmat_
    return lr

def hmm_fit_predict(hmmModel, X, lengths=None, verbose=0):
    hmmModel.fit(X, lengths=lengths)
    predictions = hmmModel.predict(X, lengths=lengths)
    if verbose>0:
        print("hmmModel.monitor_:")
        print(hmmModel.monitor_)
        print("hmmModel.converged:")
        print(hmmModel.converged)
        print(predictions)
    return predictions
