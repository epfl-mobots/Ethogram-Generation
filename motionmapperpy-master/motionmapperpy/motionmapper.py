import os, time, glob, shutil
import multiprocessing as mp

import matplotlib
matplotlib.use('Agg')

import numpy as np
from scipy.io import savemat, loadmat
from sklearn.manifold import TSNE
import hdf5storage
from sklearn.neighbors import NearestNeighbors
from skimage.segmentation import watershed
import h5py
from easydict import EasyDict as edict
from scipy.spatial import Delaunay, distance
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from skimage.filters import roberts

from .wavelet import findWavelets
from .mmutils import findPointDensity, gencmap
from .setrunparameters import setRunParameters
from umap import UMAP
import pickle

"""Core t-SNE MotionMapper functions."""

def findKLDivergences(data):
    N = len(data)
    logData = np.log(data)
    logData[~np.isfinite(logData)] = 0

    entropies = -np.sum(np.multiply(data, logData), 1)

    D = - np.dot(data, logData.T)

    D = D - entropies[:,None]

    D = D / np.log(2)
    np.fill_diagonal(D, 0)
    return D, entropies

def run_UMAP(data, parameters, save_model=True):
    if not parameters.waveletDecomp:
        raise ValueError('UMAP not implemented without wavelet decomposition.')

    vals = np.sum(data, 1)
    if ~np.all(vals == 1):
        data = data / vals[:, None]

    umapfolder = parameters['projectPath'] + '/UMAP/'
    n_neighbors, train_negative_sample_rate, min_dist, umap_output_dims, n_training_epochs = parameters['n_neighbors'], \
                                            parameters['train_negative_sample_rate'], parameters['min_dist'], \
                                            parameters['umap_output_dims'], parameters['n_training_epochs']

    um = UMAP(n_neighbors=n_neighbors, negative_sample_rate=train_negative_sample_rate, min_dist=min_dist,
              n_components=umap_output_dims, n_epochs=n_training_epochs)
    y = um.fit_transform(data)
    trainmean = np.mean(y, 0)
    scale = (parameters['rescale_max']/np.abs(y).max())
    y = y - trainmean
    y = y * scale

    if save_model:
        print('Saving UMAP model to disk...')
        np.save(umapfolder+'_trainMeanScale.npy', np.array([trainmean, scale], dtype=object))
        with open(umapfolder+'umap.model', 'wb') as f:
            pickle.dump(um, f)

    return y

def run_tSne(data, parameters=None):
    """
    run_tSne runs the t-SNE algorithm on an array of normalized wavelet amplitudes
    :param data: Nxd array of wavelet amplitudes (will normalize if unnormalized) containing N data points
    :param parameters: motionmapperpy Parameters dictionary.
    :return:
            yData -> N x 2 array of embedding results
    """
    parameters = setRunParameters(parameters)

    vals = np.sum(data, 1)
    if ~np.all(vals == 1):
        data = data / vals[:, None]

    if parameters.waveletDecomp:
        print('Finding Distances')
        D, _ = findKLDivergences(data)
        D[~np.isfinite(D)] = 0.0
        D = np.square(D)

        print('Computing t-SNE with %s method'%parameters.tSNE_method)
        tsne = TSNE(perplexity=parameters.perplexity, metric='precomputed', verbose=1, n_jobs=-1,
                    method=parameters.tSNE_method, init='random')
        yData = tsne.fit_transform(D)
    else:
        tsne = TSNE(perplexity=parameters.perplexity, metric='euclidean', verbose=1, n_jobs=-1,
                    method=parameters.tSNE_method)
        yData = tsne.fit_transform(data)
        # raise ValueError('tSNE not implemented for runs without wavelet decomposition.')
    return yData


"""Training-set Generation"""


def returnTemplates(yData, signalData, minTemplateLength=10, kdNeighbors=10):
    maxY = np.ceil(np.max(np.abs(yData[:]))) + 1
    d = signalData.shape[1]

    nn = NearestNeighbors(n_neighbors=kdNeighbors + 1, n_jobs=-1)
    nn.fit(yData)
    D, _ = nn.kneighbors(yData)
    sigma = np.median(D[:, -1])

    _, xx, density = findPointDensity(yData, sigma, 501, [-maxY, maxY])

    L = watershed(-density, connectivity=10)

    # savemat('/mnt/HFSP_Data/scripts/LIDAR/testdata.mat', {'ydata':yData, 'D':D, 'density':density, 'L':L})

    watershedValues = np.digitize(yData, xx)
    watershedValues = L[watershedValues[:, 1], watershedValues[:, 0]]

    maxL = np.max(L)

    templates = []
    for i in range(1, maxL + 1):
        templates.append(signalData[watershedValues == i])
    lengths = np.array([len(i) for i in templates])
    templates = np.array(templates, dtype=object)

    idx = np.where(lengths >= minTemplateLength)[0]
    vals2 = np.zeros(watershedValues.shape)
    for i in range(len(idx)):
        vals2[watershedValues == idx[i]+1] = i + 1

    templates = templates[lengths >= minTemplateLength]
    lengths = lengths[lengths >= minTemplateLength]

    return templates, xx, density, sigma, lengths, L, vals2


def findTemplatesFromData(signalData, yData, signalAmps, numPerDataSet, parameters,projectionFile):
    kdNeighbors = parameters.kdNeighbors
    minTemplateLength = parameters.minTemplateLength

    print('Finding Templates.')
    templates, _, density, _, templateLengths, L, vals = returnTemplates(yData, signalData, minTemplateLength, kdNeighbors)

    ####################################################
    wbounds = np.where(roberts(L).astype('bool'))
    wbounds = (wbounds[1], wbounds[0])
    fig, ax = plt.subplots()
    ax.imshow(density, origin='lower', cmap=gencmap())
    ax.scatter(wbounds[0], wbounds[1], color='k', s=0.1)
    fig.savefig(projectionFile[:-4]+'_trainingtSNE.png')
    plt.close()
    ####################################################

    N = len(templates)
    d = len(signalData[1, :])
    selectedData = np.zeros((numPerDataSet, d))
    selectedAmps = np.zeros((numPerDataSet, 1))

    numInGroup = np.round(numPerDataSet * templateLengths / np.sum(templateLengths))
    numInGroup[numInGroup == 0] = 1
    sumVal = np.sum(numInGroup)
    if sumVal < numPerDataSet:
        q = int(numPerDataSet - sumVal)
        idx = np.random.permutation(N)[:min(q, N)]
        numInGroup[idx] = numInGroup[idx] + 1
    else:
        if sumVal > numPerDataSet:
            q = int(sumVal - numPerDataSet)
            idx2 = np.where(numInGroup > 1)[0]
            Lq = len(idx2)
            if Lq < q:
                idx2 = np.arange(len(numInGroup))
            idx = np.random.permutation(len(idx2))[:q]
            numInGroup[idx2[idx]] = numInGroup[idx2[idx]] - 1
    idx = numInGroup > templateLengths
    numInGroup[idx] = templateLengths[idx]
    cumSumGroupVals = [0] + np.cumsum(numInGroup).astype(int).tolist()

    for j in range(N):

        if cumSumGroupVals[j + 1] > cumSumGroupVals[j]:
            amps = signalAmps[vals == j+1]
            idx2 = np.random.permutation(len(templates[j][:, 1]))[:int(numInGroup[j])].astype(int)
            selectedData[cumSumGroupVals[j]:cumSumGroupVals[j + 1], :] = templates[j][idx2, :]
            selectedAmps[cumSumGroupVals[j]:cumSumGroupVals[j + 1], 0] = amps[idx2]

    signalData = selectedData
    signalAmps = selectedAmps

    return signalData, signalAmps

def mm_findWavelets(projections, numModes, parameters):

    amplitudes, f = findWavelets(projections, numModes, parameters.omega0, parameters.numPeriods,
                                 parameters.samplingFreq, parameters.maxF, parameters.minF, parameters.numProcessors,
                                 parameters.useGPU)
    return amplitudes, f


def _normalize_optional_vector(vector, expected_length):
    if vector is None:
        return None

    array = np.asarray(vector).squeeze()
    if array.ndim == 0:
        array = array.reshape(1)
    if len(array) != expected_length:
        raise ValueError('Metadata length does not match projections length.')
    return array


def _chunk_bounds_from_metadata(length, parameters, timestamps=None, chunk_ids=None):
    breakpoints = set()
    chunk_ids = _normalize_optional_vector(chunk_ids, length)
    if chunk_ids is not None:
        breakpoints.update((np.flatnonzero(chunk_ids[1:] != chunk_ids[:-1]) + 1).tolist())

    timestamps = _normalize_optional_vector(timestamps, length)
    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=float)
        diffs = np.diff(ts)
        finite_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if finite_diffs.size > 0:
            expected_dt = np.median(finite_diffs)
            gap_multiplier = getattr(parameters, 'waveletGapThresholdMultiplier')
            gap_threshold = expected_dt * gap_multiplier
            breakpoints.update((np.flatnonzero(diffs > gap_threshold) + 1).tolist())

    if not breakpoints:
        return [(0, length)]

    breakpoints = sorted(bp for bp in breakpoints if 0 < bp < length)
    starts = [0] + breakpoints
    ends = breakpoints + [length]
    return list(zip(starts, ends))


def _wavelet_edge_trim_samples(parameters):
    custom_trim = getattr(parameters, 'waveletEdgeTrimSamples', None)
    if custom_trim is not None:
        return max(0, int(custom_trim))

    samples_per_period = int(np.ceil((1.0 / parameters.minF) * parameters.samplingFreq))
    return max(1, samples_per_period // 2)


def mm_findWavelets_with_chunks(projections, numModes, parameters, timestamps=None, chunk_ids=None, return_report=True):
    projections = np.asarray(projections)
    length = projections.shape[0]
    chunk_bounds = _chunk_bounds_from_metadata(length, parameters, timestamps=timestamps, chunk_ids=chunk_ids)

    if len(chunk_bounds) == 1 and chunk_bounds[0] == (0, length):
        amplitudes, f = findWavelets(projections, numModes, parameters.omega0, parameters.numPeriods,
                                     parameters.samplingFreq, parameters.maxF, parameters.minF,
                                     parameters.numProcessors, parameters.useGPU)
        if hasattr(amplitudes, 'get'):
            amplitudes = amplitudes.get()
        kept_indices = np.arange(length, dtype=int)
        report = {
            'total_samples': int(length),
            'kept_samples': int(len(kept_indices)),
            'skipped_samples': 0,
            'usable_fraction': 1.0,
            'usable_percentage': 100.0,
            'retained_chunks': 1,
            'skipped_chunks': [],
        }
        if return_report:
            return amplitudes, f, kept_indices, report
        return amplitudes, f, kept_indices

    retained_amplitudes = []
    retained_indices = []
    skipped_chunks = []
    f = None
    edge_trim = _wavelet_edge_trim_samples(parameters)
    min_chunk_len = parameters.samplingFreq/parameters.minF # minimum chunk length in samples

    for start, end in chunk_bounds:
        chunk = projections[start:end]
        if len(chunk) <= 2 * edge_trim or len(chunk) < min_chunk_len:
            skipped_chunks.append((start, end, len(chunk)))
            continue

        chunk_amplitudes, f = findWavelets(chunk, numModes, parameters.omega0, parameters.numPeriods,
                                           parameters.samplingFreq, parameters.maxF, parameters.minF,
                                           parameters.numProcessors, parameters.useGPU)
        if hasattr(chunk_amplitudes, 'get'):
            chunk_amplitudes = chunk_amplitudes.get()

        kept = slice(edge_trim, len(chunk) - edge_trim)
        retained_amplitudes.append(chunk_amplitudes[kept])
        retained_indices.append(np.arange(start, end, dtype=int)[kept])

    if not retained_amplitudes:
        report = {
            'total_samples': int(length),
            'kept_samples': 0,
            'skipped_samples': int(length),
            'usable_fraction': 0.0,
            'usable_percentage': 0.0,
            'retained_chunks': 0,
            'skipped_chunks': skipped_chunks,
        }
        print(
            f"Skipping all wavelet chunks: 0/{length} samples retained (0.00%). "
            f"Edge trim={edge_trim}. Skipped chunks: {len(skipped_chunks)}"
        )
        if return_report:
            return np.empty((0, numModes * parameters.numPeriods)), f, np.empty((0,), dtype=int), report
        raise ValueError('No wavelet chunks were long enough after edge trimming.')

    amplitudes = np.concatenate(retained_amplitudes, axis=0)
    kept_indices = np.concatenate(retained_indices, axis=0)
    report = {
        'total_samples': int(length),
        'kept_samples': int(len(kept_indices)),
        'skipped_samples': int(length - len(kept_indices)),
        'usable_fraction': float(len(kept_indices) / length) if length else 0.0,
        'usable_percentage': float(100.0 * len(kept_indices) / length) if length else 0.0,
        'retained_chunks': len(retained_amplitudes),
        'skipped_chunks': skipped_chunks,
    }
    if skipped_chunks:
        skipped_samples = sum(chunk_length for _, _, chunk_length in skipped_chunks)
        print(
            f"Wavelet chunk report: kept {len(kept_indices)}/{length} samples "
            f"({report['usable_percentage']:.2f}%). "
            f"Retained chunks: {len(retained_amplitudes)}, skipped chunks: {len(skipped_chunks)}, "
            f"skipped samples: {skipped_samples}."
        )
    if return_report:
        return amplitudes, f, kept_indices, report
    return amplitudes, f, kept_indices

def file_embeddingSubSampling(projectionFile, parameters):
    perplexity = parameters.training_perplexity
    numPoints = parameters.training_numPoints

    print('\t Loading Projections')
    try:
        projection_mat = loadmat(projectionFile)
        projections = np.array(projection_mat['projections'])
        timestamps = projection_mat.get('real_timestamps')
    except:
        with h5py.File(projectionFile, 'r') as hfile:
            projections = hfile['projections'][:].T
            timestamps = hfile['real_timestamps'][:] if 'real_timestamps' in hfile else None
        projections = np.array(projections)

    timestamps = _normalize_optional_vector(timestamps, len(projections))

    if projections.shape[0] < numPoints:
        raise ValueError('Training number of points for miniTSNE is greater than # samples in some files. Please '
                         'adjust it to %i or lower'%(projections.shape[0]))

    N = len(projections)
    numModes = parameters.pcaModes
    skipLength = np.floor(N / numPoints).astype(int)
    if skipLength == 0:
        skipLength = 1
        numPoints = N

    firstFrame = (N%numPoints)

    if parameters.waveletDecomp:
        print('\t Calculating Wavelets')
        data, _, signalIdx, report = mm_findWavelets_with_chunks(
            projections,
            numModes,
            parameters,
            timestamps=timestamps,
            return_report=True,
        )
        print(
            f"\t Wavelet coverage: {report['kept_samples']}/{report['total_samples']} samples "
            f"({report['usable_percentage']:.2f}%) retained after chunk trimming."
        )
        if len(signalIdx) == 0:
            raise ValueError('No wavelet samples were retained after chunk trimming.')
        # 'signalIdx' returned from mm_findWavelets_with_chunks are original projection indices
        # but 'data' contains only the retained wavelet rows in the same order. To index
        # into 'data' we must use positions within the kept indices array rather than
        # the original projection indices.
        kept_indices = signalIdx
        # build a mapping from original projection index -> row position in 'data'
        pos_map = -np.ones((len(projections),), dtype=int)
        pos_map[kept_indices] = np.arange(len(kept_indices), dtype=int)
        # compute safe slice endpoint to avoid selecting beyond retained rows
        end_pos = int(firstFrame + (numPoints) * skipLength)
        end_pos = min(end_pos, data.shape[0])
        if end_pos <= firstFrame:
            raise ValueError('Computed sampling slice is empty (end_pos <= firstFrame).')
        # sample positions within the retained-wavelet rows directly (safer)
        pos_positions = np.arange(data.shape[0], dtype=int)[firstFrame:end_pos: skipLength]
        selected_orig = kept_indices[pos_positions]
        posIdx = pos_positions
        # clip positions to data length just in case
        pos_mask = (posIdx >= 0) & (posIdx < data.shape[0])
        if not np.all(pos_mask):
            bad_orig = selected_orig[~pos_mask]
            print(f'Warning: {bad_orig.size} selected original indices map outside retained wavelet rows. Dropping them: {bad_orig[:10]}')
        posIdx = posIdx[pos_mask]
        if posIdx.size == 0:
            raise ValueError('No valid signal indices remain after alignment and bounds-checking.')
        # also provide signalIdx as the original projection indices for downstream use
        # diagnostic info 
        signalIdx = selected_orig
        print(f'DEBUG wavelet: N={len(projections)}, kept={len(kept_indices)}, data_rows={data.shape[0]}, '
            f'firstFrame={firstFrame}, skipLength={skipLength}, numPoints={numPoints}, end_pos={end_pos}, '
            f'selected={len(selected_orig)}, kept_selected={len(signalIdx)}')
        if posIdx.size:
            print(f'DEBUG pos_positions range: min={int(posIdx.min())}, max={int(posIdx.max())}')
        signalData = data[posIdx]
    else:
        print('Using projections for tSNE. No wavelet decomposition.')
        data = projections
        signalIdx = np.indices((data.shape[0],))[0]
        end_pos = int(firstFrame + (numPoints) * skipLength)
        end_pos = min(end_pos, data.shape[0])
        if end_pos <= firstFrame:
            raise ValueError('Computed sampling slice is empty (end_pos <= firstFrame).')
        signalIdx = signalIdx[firstFrame:end_pos: skipLength]
        # signalIdx here are simple integer positions into 'data'
        signalIdx = signalIdx[(signalIdx >= 0) & (signalIdx < data.shape[0])]
        if signalIdx.size == 0:
            raise ValueError('No valid signal indices remain after alignment and bounds-checking.')
        print(f'DEBUG projections-branch: data_rows={data.shape[0]}, firstFrame={firstFrame}, skipLength={skipLength}, numPoints={numPoints}, selected={len(signalIdx)}')
        signalData = data[signalIdx]

    signalAmps = np.sum(signalData, axis=1)

    signalData = signalData/signalAmps[:,None]

    if parameters.method == 'TSNE':
        parameters.perplexity = perplexity
        yData = run_tSne(signalData, parameters)
    elif parameters.method == 'UMAP':
        yData = run_UMAP(signalData, parameters, save_model=False)
    else:
        raise ValueError('Supported parameter.method are \'TSNE\' or \'UMAP\'')
    return yData, signalData, signalIdx, signalAmps

def runEmbeddingSubSampling(projectionDirectory, parameters):
    """
    runEmbeddingSubSampling generates a training set given a set of .mat files.

    :param projectionDirectory: directory path containing .mat projection files.
    Each of these files should contain an N x pcaModes variable, 'projections'.
    :param parameters: motionmapperpy Parameters dictionary.
    :return:
        trainingSetData -> normalized wavelet training set
                           (N x (pcaModes*numPeriods) )
        trainingSetAmps -> Nx1 array of training set wavelet amplitudes
        projectionFiles -> list of files in 'projectionDirectory'
    """
    parameters = setRunParameters(parameters)
    projectionFiles = glob.glob(projectionDirectory+'/*pcaModes.mat')
    
    N = parameters.trainingSetSize
    L = len(projectionFiles)
    assert L > 0, "No projection files found in directory: %s" % projectionDirectory
    numPerDataSet = round(N / L)
    numModes = parameters.pcaModes
    numPeriods = parameters.numPeriods

    if numPerDataSet > parameters.training_numPoints:
        raise ValueError("miniTSNE size is %i samples per file which is low for current trainingSetSize which "
                         "requries %i samples per file. "
                         "Please decrease trainingSetSize or increase training_numPoints."%
                         (parameters.training_numPoints, numPerDataSet))

    if parameters.waveletDecomp:
        trainingSetData = np.zeros((numPerDataSet * L, numModes * numPeriods))
    else:
        trainingSetData = np.zeros((numPerDataSet * L, numModes))
    trainingSetAmps = np.zeros((numPerDataSet * L, 1))
    useIdx = np.ones((numPerDataSet * L), dtype='bool')

    for i in range(L):

        print('Finding training set contributions from data set %i/%i : \n%s'%(i+1, L, projectionFiles[i]))

        currentIdx = np.arange(numPerDataSet) + (i * numPerDataSet)

        yData, signalData, _, signalAmps = file_embeddingSubSampling(projectionFiles[i], parameters)

        trainingSetData[currentIdx,:], trainingSetAmps[currentIdx] = findTemplatesFromData(signalData, yData,
                                                                                           signalAmps, numPerDataSet,
                                                                                        parameters,projectionFiles[i])

        a = (np.sum(trainingSetData[currentIdx,:], 1) == 0)
        useIdx[currentIdx[a]] = False

    trainingSetData = trainingSetData[useIdx,:]
    trainingSetAmps = trainingSetAmps[useIdx]

    return trainingSetData, trainingSetAmps, projectionFiles

def subsampled_tsne_from_projections(parameters):
    """
    Wrapper function for training set subsampling and mapping.
    """
    results_directory = parameters.projectPath
    projection_directory = results_directory+'/Projections/'
    if parameters.method == 'TSNE':
        if parameters.waveletDecomp:
            tsne_directory= results_directory+'/TSNE/'
        else:
            tsne_directory = results_directory + '/TSNE_Projections/'

        parameters.tsne_directory = tsne_directory

        parameters.tsne_readout = 50

        tSNE_method_old = parameters.tSNE_method
        if tSNE_method_old  != 'barnes_hut':
            print('Setting tsne method to barnes_hut while subsampling for training set (for speedup)...')
            parameters.tSNE_method = 'barnes_hut'

    elif parameters.method == 'UMAP':
        tsne_directory = results_directory + '/UMAP/'
        if not parameters.waveletDecomp:
            raise ValueError('Wavelet decomposition needed to run UMAP implementation.')
    else:
        raise ValueError('Supported parameter.method are \'TSNE\' or \'UMAP\'')

    print('Finding Training Set')
    if not os.path.exists(tsne_directory+'training_data.mat'):
        trainingSetData, trainingSetAmps,_ = runEmbeddingSubSampling(projection_directory, parameters)
        if os.path.exists(tsne_directory):
            shutil.rmtree(tsne_directory)
            os.mkdir(tsne_directory)
        else:
            os.mkdir(tsne_directory)

        hdf5storage.write(data={'trainingSetData': trainingSetData}, path='/', truncate_existing=True,
                          filename=tsne_directory+'/training_data.mat', store_python_metadata=False,
                          matlab_compatible=True)

        hdf5storage.write(data={'trainingSetAmps': trainingSetAmps}, path='/', truncate_existing=True,
                          filename=tsne_directory + '/training_amps.mat', store_python_metadata=False,
                          matlab_compatible=True)


        del trainingSetAmps
    else:
        print('Subsampled trainingSetData found, skipping minitSNE and running training tSNE')
        with h5py.File(tsne_directory + '/training_data.mat', 'r') as hfile:
            trainingSetData = hfile['trainingSetData'][:].T


    # %% Run t-SNE on training set
    if parameters.method == 'TSNE':
        if tSNE_method_old  != 'barnes_hut':
            print('Setting tsne method back to to %s' % tSNE_method_old)
            parameters.tSNE_method = tSNE_method_old
        parameters.tsne_readout = 5
        print('Finding t-SNE Embedding for Training Set')

        trainingEmbedding= run_tSne(trainingSetData,parameters)
    elif parameters.method == 'UMAP':
        print('Finding UMAP Embedding for Training Set')
        trainingEmbedding = run_UMAP(trainingSetData, parameters)
    else:
        raise ValueError('Supported parameter.method are \'TSNE\' or \'UMAP\'')
    hdf5storage.write(data={'trainingEmbedding': trainingEmbedding}, path='/', truncate_existing=True,
                      filename=tsne_directory + '/training_embedding.mat', store_python_metadata=False,
                      matlab_compatible=True)


"""Re-Embedding Code"""


def returnCorrectSigma_sparse(ds, perplexity, tol,maxNeighbors):

    highGuess = np.max(ds)
    lowGuess = 1e-10

    sigma = .5*(highGuess + lowGuess)

    dsize = ds.shape
    sortIdx = np.argsort(ds)
    ds = ds[sortIdx[:maxNeighbors]]
    p = np.exp(-0.5*np.square(ds)/sigma**2)
    p = p/np.sum(p)
    idx = p>0
    H = np.sum(-np.multiply(p[idx],np.log(p[idx]))/np.log(2))
    P = 2**H

    if abs(P-perplexity) < tol:
        test = False
    else:
        test = True

    count = 0
    if ~np.isfinite(sigma):
        raise ValueError('Starting sigma is %0.02f, highGuess is %0.02f '
                'and lowGuess is %0.02f'%(sigma, highGuess, lowGuess))
    while test:

        if P > perplexity:
            highGuess = sigma
        else:
            lowGuess = sigma

        sigma = .5*(highGuess + lowGuess)


        p = np.exp(-.5*np.square(ds)/sigma**2)
        if np.sum(p) > 0:
            p = p/np.sum(p)
        idx = p>0
        H = np.sum(-np.multiply(p[idx],np.log(p[idx]))/np.log(2))
        P = 2**H

        if np.abs(P-perplexity) < tol:
            test = False

    out = np.zeros((dsize[0],))
    out[sortIdx[:maxNeighbors]] = p
    return sigma,out


def findListKLDivergences(data, data2):
    logData = np.log(data)

    entropies = -np.sum(np.multiply(data,logData), 1)
    del logData

    logData2 = np.log(data2)

    D = - np.dot(data,logData2.T)

    D = D - entropies[:,None]

    D = D / np.log(2)
    return D,entropies


def calculateKLCost(x,ydata,ps):
    d = np.sum(np.square(ydata-x),1).T
    out = np.log(np.sum(1/(1+d))) + np.sum(np.multiply(ps,np.log(1+d)))
    return out


def TDistProjs(i, q, perplexity, sigmaTolerance, maxNeighbors, trainingEmbedding, readout, waveletDecomp):
    if (i+1)%readout == 0:
        t1 = time.time()
        print('\t\t Calculating Sigma Image #%5i'% (i+1))
    _, p = returnCorrectSigma_sparse(q, perplexity, sigmaTolerance, maxNeighbors)

    if (i+1)%readout == 0:
        print('\t\t Calculated Sigma Image #%5i'%(i+1))

    idx2 = p>0
    z = trainingEmbedding[idx2,:]
    maxIdx = np.argmax(p)
    a = np.sum(z*(p[idx2].T)[:,None],axis=0)

    guesses = [a, trainingEmbedding[maxIdx,:]]

    q = Delaunay(z)

    if (i+1)%readout == 0:
        print('\t\t FminSearch Image #%5i'%(i+1))

    b = np.zeros((2, 2))
    c = np.zeros((2,))
    flags = np.zeros((2,))

    if waveletDecomp:
        costfunc = calculateKLCost
    else:
        costfunc = calculateKLCost

    b[0, :], c[0], _, _, flags[0] = fmin(costfunc, x0=guesses[0], args=(z, p[idx2]), disp=False,
                                         full_output=True, maxiter=100)
    b[1, :], c[1], _, _, flags[1] = fmin(costfunc, x0=guesses[1], args=(z, p[idx2]), disp=False,
                                         full_output=True, maxiter=100)
    if (i+1)%readout == 0:
        print('\t\t FminSearch Done Image #%5i %0.02fseconds flags are %s'%(i+1, time.time()-t1, flags))

    polyIn = q.find_simplex(b)>=0

    if np.sum(polyIn) > 0:
        pp = np.where(polyIn)[0]
        mI = np.argmin(c[polyIn])
        mI = pp[mI]
        current_poly = True
    else:
        mI = np.argmin(c)
        current_poly = False
    if (i+1)%readout == 0:
        print('\t\t Simplex search done Image #%5i %0.02fseconds'%(i+1, time.time()-t1))
    exitFlags = flags[mI]
    current_guesses = guesses[mI]
    current = b[mI]
    tCosts = c[mI]
    current_meanMax = mI
    return current_guesses, current, tCosts, current_poly, current_meanMax, exitFlags


def findTDistributedProjections_fmin(data, trainingData, trainingEmbedding, parameters):
    readout = 20000
    sigmaTolerance = 1e-5
    perplexity = parameters.perplexity
    maxNeighbors = parameters.maxNeighbors
    batchSize = parameters.embedding_batchSize



    N = len(data)
    zValues = np.zeros((N,2))
    zGuesses = np.zeros((N,2))
    zCosts = np.zeros((N,))
    batches = np.ceil(N/batchSize).astype(int)
    inConvHull = np.zeros((N,), dtype=bool)
    meanMax = np.zeros((N,))
    exitFlags = np.zeros((N,))

    if parameters.numProcessors < 0:
        numProcessors = mp.cpu_count()
    else:
        numProcessors = parameters.numProcessors
    # ctx = mp.get_context('spawn')

    for j in range(batches):
        print('\t Processing batch #%4i out of %4i'%(j+1,batches))
        idx = np.arange(batchSize) + j*batchSize
        idx = idx[idx < N]
        currentData = data[idx,:]

        if parameters.waveletDecomp:
            if np.sum(currentData==0):
                print('Zeros found in wavelet data at following positions. Will replace then with 1e-12.')
                currentData[currentData==0] = 1e-12

            print('\t Calculating distances for batch %4i'%(j+1))
            t1 = time.time()
            D2,_ = findListKLDivergences(currentData,trainingData)
            print('\t Calculated distances for batch %4i %0.02fseconds.'%(j+1, time.time()-t1))
        else:
            print('\t Calculating distances for batch %4i' % (j + 1))
            t1 = time.time()
            D2 = distance.cdist(currentData, trainingData, metric='sqeuclidean')
            print('\t Calculated distances for batch %4i %0.02fseconds.' % (j + 1, time.time() - t1))

        print('\t Calculating fminProjections for batch %4i' % (j + 1))
        t1 = time.time()
        pool = mp.Pool(numProcessors)
        outs = pool.starmap(TDistProjs, [(i, D2[i,:], perplexity, sigmaTolerance, maxNeighbors, trainingEmbedding, readout, parameters.waveletDecomp)
                            for i in range(len(idx))])

        zGuesses[idx,:] = np.concatenate([out[0][:,None] for out in outs], axis=1).T
        zValues[idx,:] = np.concatenate([out[1][:,None] for out in outs], axis=1).T
        zCosts[idx] = np.array([out[2] for out in outs])
        inConvHull[idx] = np.array([out[3] for out in outs])
        meanMax[idx] = np.array([out[4] for out in outs])
        exitFlags[idx] = np.array([out[5] for out in outs])
        pool.close()
        pool.join()
        print('\t Processed batch #%4i out of %4i in %0.02fseconds.\n'%(j+1, batches, time.time()-t1))

    zValues[~inConvHull,:] = zGuesses[~inConvHull,:]

    return zValues,zCosts,zGuesses,inConvHull,meanMax,exitFlags


def findEmbeddings(projections, trainingData, trainingEmbedding, parameters, timestamps=None, chunk_ids=None):
    """
    findEmbeddings finds the optimal embedding of a data set into a previously
    found t-SNE embedding.
    :param projections:  N x (pcaModes x numPeriods) array of projection values.
    :param trainingData: Nt x (pcaModes x numPeriods) array of wavelet amplitudes containing Nt data points.
    :param trainingEmbedding: Nt x 2 array of embeddings.
    :param parameters: motionmapperpy Parameters dictionary.
    :param timestamps: Optional timestamps used to split wavelet computation into contiguous chunks.
    :param chunk_ids: Optional chunk labels used to split wavelet computation into contiguous chunks.
    :return: zValues : N x 2 array of embedding results, outputStatistics : dictionary containing other parametric
    outputs.
    """
    d = projections.shape[1]
    numModes = parameters.pcaModes
    numPeriods = parameters.numPeriods

    if parameters.waveletDecomp:
        print('Finding Wavelets')
        data, f, keptIdx, report = mm_findWavelets_with_chunks(
            projections,
            numModes,
            parameters,
            timestamps=timestamps,
            chunk_ids=chunk_ids
        )
        print(
            f"Wavelet coverage: {report['kept_samples']}/{report['total_samples']} samples "
            f"({report['usable_percentage']:.2f}%) retained after chunk trimming."
        )
        if len(keptIdx) == 0:
            raise ValueError('No wavelet samples were retained after chunk trimming.')
    else:
        print('Using projections for tSNE. No wavelet decomposition.')
        f = 0
        keptIdx = np.arange(len(projections), dtype=int)
        report = {
            'total_samples': int(len(projections)),
            'kept_samples': int(len(projections)),
            'skipped_samples': 0,
            'usable_fraction': 1.0,
            'usable_percentage': 100.0,
            'retained_chunks': 1,
            'skipped_chunks': [],
        }
        data = projections
    data = data / np.sum(data, 1)[:, None]

    print('Finding Embeddings')
    t1 = time.time()
    if parameters.method == 'TSNE':
        zValues, zCosts, zGuesses, inConvHull, meanMax, exitFlags = findTDistributedProjections_fmin(data,
                                                                                trainingData, trainingEmbedding, parameters)

        outputStatistics = edict()
        outputStatistics.zCosts = zCosts
        outputStatistics.f = f
        outputStatistics.numModes = numModes
        outputStatistics.zGuesses = zGuesses
        outputStatistics.inConvHull = inConvHull
        outputStatistics.meanMax = meanMax
        outputStatistics.exitFlags = exitFlags
        outputStatistics.keptIdx = keptIdx
        outputStatistics.waveletCoverage = report
    elif parameters.method == 'UMAP':
        umapfolder = parameters['projectPath'] + '/UMAP/'
        print('\tLoading UMAP Model.')
        with open(umapfolder + 'umap.model', 'rb') as f:
            um = pickle.load(f)
        trainparams = np.load(umapfolder + '_trainMeanScale.npy', allow_pickle=True)
        print('\tLoaded.')
        embed_negative_sample_rate = parameters['embed_negative_sample_rate']
        um.negative_sample_rate = embed_negative_sample_rate
        zValues = um.transform(data)
        zValues = zValues - trainparams[0]
        zValues = zValues * trainparams[1]
        outputStatistics = edict()
        outputStatistics.training_mean = trainparams[0]
        outputStatistics.training_scale = trainparams[1]
        outputStatistics.keptIdx = keptIdx
        outputStatistics.waveletCoverage = report
    else:
        raise ValueError('Supported parameter.method are \'TSNE\' or \'UMAP\'')
    del data
    print('Embeddings found in %0.02f seconds.'%(time.time()-t1))

    return zValues,outputStatistics

