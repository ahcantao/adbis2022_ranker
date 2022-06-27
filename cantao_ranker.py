import ast #converts string to list
import pickle
import glob
import math
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing
number_of_cores =  multiprocessing.cpu_count()

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.forest import _generate_unsampled_indices 

import networkx as nx

def create_output_files(output_path):
    output_header = list()
    output_header.append(['orientation', 'centrality', 'centrality_or_forest',
                          'ranking_score',
                          'dataset', 'noise', 'R', 'features_score'])
    output_header = pd.DataFrame(output_header)
    for i in range(1,11):
        output_header.to_csv(f'{output_path}/table_of_scores_full_{i}.csv', header=False, index=None, sep=',', mode='w')

def save_to_files(output_data,output_path,kfold):
    output_data = pd.DataFrame(output_data)
    output_data.to_csv(f'{output_path}/table_of_scores_full_{kfold}.csv', header=False, index=None, sep=',', mode='a')

def GetnImportantFeatures(dataset_name):
    if dataset_name in datasets_d2:
        nImportantFeatures = 2
    elif dataset_name in datasets_d3:
        nImportantFeatures = 3
    elif dataset_name in datasets_d5:
        nImportantFeatures = 5
    elif dataset_name in datasets_d7:
        nImportantFeatures = 7
    elif dataset_name in datasets_d21:
        nImportantFeatures = 21
    else:
        print("could not count nImportantFeatures. Please check the dataset name or this message's condition.")
    return nImportantFeatures

def get_number_of_classes(dataset_name):
    if dataset_name in datasets_c2:
        n_classes = 2
    elif dataset_name in datasets_c3:
        n_classes = 3
    elif dataset_name in datasets_c4:
        n_classes = 4
    elif dataset_name in datasets_c5:
        n_classes = 5
    elif dataset_name in datasets_c7:
        n_classes = 7
    elif dataset_name in datasets_c8:
        n_classes = 8
    elif dataset_name in datasets_regression:
        n_classes = 0
    else:
        print("could not count nImportantFeatures. Please check the dataset name or this message's condition.")
    return n_classes

def generate_graph(treeWeights, totalFeatures, orientation):
    if orientation == 'in':
        """creating directed IN-graph"""
        graph = nx.DiGraph() #in-edges
        graph.add_nodes_from(range(totalFeatures))

        """inserting the edges / links / connections """
        for graphEdgesWeightMetric, treeEdgesList in treeWeights.items():
            for treeEdges in treeEdgesList:
                for (u,v,w) in treeEdges:
                    if graph.has_edge(u,v):
                        graph[u][v]['weight'] += w
                    else:
                        graph.add_edge(u, v, weight = w)
    
    elif orientation == 'out':
        """creating directed OUT-graph"""
        graph = nx.DiGraph() #out-edges
        graph.add_nodes_from(range(totalFeatures))

        for graphEdgesWeightMetric, treeEdgesList in treeWeights.items():
            for treeEdges in treeEdgesList:
                for (u,v,w) in treeEdges:
                    if graph.has_edge(v,u):
                        graph[v][u]['weight'] += w
                    else:
                        graph.add_edge(v, u, weight = w)
                        
    elif orientation == 'g':
        """creating NOT directed graph (simple graph) """
        graph = nx.Graph() #undirected graph
        graph.add_nodes_from(range(totalFeatures))

        for graphEdgesWeightMetric, treeEdgesList in treeWeights.items():
            for treeEdges in treeEdgesList:
                for (u,v,w) in treeEdges:
                    if graph.has_edge(u,v):
                        graph[u][v]['weight'] += w
                    else:
                        graph.add_edge(u, v, weight = w)
                        
    else:
        print('wrong orientation value on generate_graph()')
        return 'orientation error'
    
    return graph

def get_metric_ranking(graph, metric):
    if metric == 'katz':
        """KATZ"""
        spectrum = max(np.abs(nx.adjacency_spectrum(graph)))
        if spectrum == 0:
            alpha = 0.1
        else:
            alpha = float((1/spectrum)* 0.7)
        centrality = nx.katz_centrality_numpy(graph, weight='weight', normalized=True, alpha=alpha, beta=1)

    
    elif metric == 'eigen':
        """EIGENVECTOR"""
        counter_iterations = 0
        max_iterations     = 500
        tolerance          = 1e-6
        converged          = False
        while not converged:
            try:
                centrality = nx.eigenvector_centrality(graph, weight='weight', max_iter=max_iterations, tol=tolerance)
                converged = True
            except:
                if counter_iterations < 2:
                    max_iterations *= 2 #double the max_iter value
                    counter_iterations += 1
                    print(f"Increasing max_iterations to: {max_iterations}")
                else:
                    tolerance *= 1e+1  #reduce the tolerance by 1 decimal
                    counter_iterations = 0
                    print(f"Increasing tolerance to: {tolerance}")

        totCent = sum(centrality.values())
        for atr in centrality:
            centrality[atr] = centrality[atr] / totCent

    
    elif metric == 'str':
        """STRENGTH"""
        
        try: #works for 'in' and 'out' orientations
            centrality = dict(graph.out_degree(weight="weight"))
        except:#works for 'g' orientation
            centrality = dict(graph.degree(weight="weight"))
            
        totCent = sum(centrality.values())
        for atr in centrality:
            centrality[atr] = centrality[atr] / totCent
        
    else:
        print('wrong metric name inserted')
        return 0
    
    ranking = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return ranking

def calculate_ranking_score(ranking,totalFeatures,nImportantFeatures,n_classes): #nImportantFeatures
    #minimum score (best case)
    minScore = ((nImportantFeatures ** 2) + nImportantFeatures) / 2

    #maximum score (worst case)
    maxScore = 0
    for i in range((totalFeatures-nImportantFeatures), totalFeatures): #not using +1 in formula 'cause index starts on zero
        maxScore += i+1

    #calculating rank_avg for features with feature_importance equals to zero (0) [not used by the forest]
    df_score_temp = pd.DataFrame(ranking)
    df_score_temp.columns = ['feature', 'feat_imp']
    df_score_temp['rank_avg'] = df_score_temp['feat_imp'].rank(method='average',ascending=False)
    ranking = df_score_temp.values.tolist()
    #------------------------------------------------------------------------------------------------------
    
    rankingPosition = [rank_avg for feat_numb, feat_imp,rank_avg in ranking if feat_numb < nImportantFeatures]
    rankingScoreSum = sum(rankingPosition)
    #If no important feature was used in the forest (not considering pre-leaf nodes)
    if rankingScoreSum == 0:
        rankingScore = 0
    else:
        #calculating RS
        rankingScore = 1 - ((rankingScoreSum - minScore) / (maxScore - minScore))
        
    return rankingScore

def build_forest(file, seed):
    dataset_data = pd.read_csv(file, sep=',', header=0)
    dataset_class = dataset_data[['CLASS']]
    dataset_class = np.ravel(dataset_class)
    dataset_data.pop('CLASS')
    n_classes = len(set(dataset_class))
    
    forest = RandomForestClassifier(n_estimators = nTrees,
                                    bootstrap = True,
                                    random_state = seed,
                                    oob_score = True,
                                    n_jobs = 1)
    forest.fit(dataset_data, dataset_class)
    return forest

def convert_tree_to_network(estimators, file_name):
    treeWeights = dict()
    treeNumber = 0
    #walking through the tree
    for treeNumber in range(len(estimators)):
        #print("treeNumber: ", treeNumber, "/63")
        n_nodes        = estimators[treeNumber].tree_.node_count
        children_left  = estimators[treeNumber].tree_.children_left
        children_right = estimators[treeNumber].tree_.children_right
        feature        = estimators[treeNumber].tree_.feature
        feature_imp    = estimators[treeNumber].feature_importances_
        impurity       = estimators[treeNumber].tree_.impurity
        threshold      = estimators[treeNumber].tree_.threshold
        node_depth     = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves      = np.zeros(shape=n_nodes, dtype=bool)
        stack          = [(0, -1)]
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node #keep this block
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
                
        singleTreeData = getPairOfNodesAndWeight(graphEdgesWeightMetric,
                                                 n_nodes,
                                                 is_leaves,
                                                 children_left,
                                                 children_right,
                                                 feature,
                                                 impurity)
        treeWeights.setdefault(graphEdgesWeightMetric,[]).append(singleTreeData)
    ###saving all nodes-edges-weights as csv file for backup
    df_treeWeights = pd.DataFrame.from_dict(treeWeights)
    df_treeWeights.to_csv(f"{output_path}/forests/forest_{file_name}.csv")
    return treeWeights

def getPairOfNodesAndWeight(graphEdgesWeightMetric, n_nodes, is_leaves, children_left,
                            children_right, feature, impurity):
    graphPairOfNodesAndWeight = list()
    if graphEdgesWeightMetric == 'binary':
        for i in range(n_nodes):
            if (is_leaves[i] == False):
                if (feature[children_left[i]] >= 0): #Condition to ignore the edge linking a leaf node (-1)
                    graphPairOfNodesAndWeight.append([feature[i],feature[children_left[i]],1])
                if (feature[children_right[i]] >= 0): #Condition to ignore the edge linking a leaf node (-1)
                    graphPairOfNodesAndWeight.append([feature[i],feature[children_right[i]],1])
#     if graphEdgesWeightMetric == '...':
        #future_work
    return graphPairOfNodesAndWeight

def ranker(kfold):
    for noise_feature in noise_features:  #1,2,4,8,16,32,64,128,256,512,1024
        time_start = pd.datetime.now()
        output = list()
        n_classes = get_number_of_classes(dataset_name)
        nImportantFeatures = GetnImportantFeatures(dataset_name)
        totalFeatures = noise_feature * nImportantFeatures + nImportantFeatures
#         print(f'R: {noise_feature}')

        for noise_percentual in noises_percentual:
            # WOKING WITH DATASETS
            file_path = f"{base_path}/{dataset_name}_noisy/{kfold:0>3}/{noise_feature:0>3}/*{noise_percentual}noi*"
            files = glob.glob(file_path) 
            seed = 0
            for file in files:
                seed += 1
                #file_name is later used to save nodes and their edges in a csv file for each forest
                try:
                    file_name = file.split('\\')[1] #removing the path on jupyter
                except:
                    file_name = file.split('/')[-1] #removing the path on terminal or linux
                file_name = file_name.split('.')[0]  #removint the extension
                forest = build_forest(file, seed)
                treeWeights = convert_tree_to_network(forest.estimators_, file_name)
            
                """ RANDOM FOREST - feature importance """
                rankingForest = [(node, imp) for node, imp in enumerate(forest.feature_importances_)]
                #sorting the ranking by importance
                rankingForest.sort(key = lambda x: x[1], reverse = True)
                ranking_score = calculate_ranking_score(rankingForest,totalFeatures,nImportantFeatures,n_classes)
                
                output.append(['rf', 'rf', 'f',
                               ranking_score,
                               dataset_name, noise_percentual,
                               noise_feature, rankingForest])
                #--------------------------------------------------

                for orientation in orientations:
                    for metric in metrics:
#                         print(f'{orientation} + {metric}')
                        graph = generate_graph(treeWeights, totalFeatures, orientation)
                        ranking = get_metric_ranking(graph, metric)
                        ranking_score = calculate_ranking_score(ranking,totalFeatures,nImportantFeatures,n_classes)

                        output.append([orientation, metric, 'c',
                                       ranking_score,
                                       dataset_name, noise_percentual,
                                       noise_feature, ranking])
        
        save_to_files(output, output_path, kfold)
        
        #dataset, R, time_running
        print(f'{dataset_name},{noise_feature},{pd.datetime.now() - time_start}')


#total d2 = 30
datasets_d2 = [ 'cassini',
                'circles',
                'moons',
                'shapes',
                'smiley',
                'spirals',
                '2dnormals_c2',
                '2dnormals_c3',
                '2dnormals_c5',
                '2dnormals_c7',
                'spirals_c2',
                'spirals_c3',
                'spirals_c5',
                'spirals_c7',
                'blobs_c2_d2',
                'blobs_c3_d2',
                'blobs_c5_d2',
                'blobs_c7_d2',
                'circle_d2',
                'hypercube_d2',
                'ringnorm_d2',
                'threenorm_d2',
                'twonorm_d2',
                'xor_d2',
                'classification_c2_d2',
                'classification_c3_d2',
                #'classification_c5_d2', #C{5,7} does not work with d=2
                #'classification_c7_d2', #C{5,7} does not work with d=2
                'gaussian_quantiles_c2_d2',
                'gaussian_quantiles_c3_d2',
                'gaussian_quantiles_c5_d2',
                'gaussian_quantiles_c7_d2']
#total d3 = 22
datasets_d3 = [ 'cuboids',
                'dinisurface',
                'helicoid',
                'swissroll',
                'circle_d3',
                'hypercube_d3', 
                'ringnorm_d3',
                'threenorm_d3',
                'twonorm_d3',
                'xor_d3',
                'blobs_c2_d3',
                'blobs_c3_d3',
                'blobs_c5_d3',
                'blobs_c7_d3',
                'classification_c2_d3',
                'classification_c3_d3',
                'classification_c5_d3',
                'classification_c7_d3',
                'gaussian_quantiles_c2_d3',
                'gaussian_quantiles_c3_d3',
                'gaussian_quantiles_c5_d3',
                'gaussian_quantiles_c7_d3']

#total d5 = 16
datasets_d5 = [ 'circle_d5',
                'ringnorm_d5',
                'threenorm_d5',
                'twonorm_d5',
                'blobs_c2_d5',
                'blobs_c3_d5',
                'blobs_c5_d5',
                'blobs_c7_d5',
                'classification_c2_d5',
                'classification_c3_d5',
                'classification_c5_d5',
                'classification_c7_d5',
                'gaussian_quantiles_c2_d5',
                'gaussian_quantiles_c3_d5',
                'gaussian_quantiles_c5_d5',
                'gaussian_quantiles_c7_d5']
#total d7 = 16
datasets_d7 = [ 'circle_d7',
                'ringnorm_d7',
                'threenorm_d7',
                'twonorm_d7',
                'blobs_c2_d7',
                'blobs_c3_d7',
                'blobs_c5_d7',
                'blobs_c7_d7',
                'classification_c2_d7',
                'classification_c3_d7',
                'classification_c5_d7',
                'classification_c7_d7',
                'gaussian_quantiles_c2_d7',
                'gaussian_quantiles_c3_d7',
                'gaussian_quantiles_c5_d7',
                'gaussian_quantiles_c7_d7']
#total d21 = 1
datasets_d21 = ['waveform']
#total c2 = 33
datasets_c2 = ['2dnormals_c2',
               'circles',
               'circle_d2',
               'circle_d3',
               'circle_d5',
               'circle_d7',
               'moons',
               'spirals',
               'spirals_c2',
               'ringnorm_d2',
               'ringnorm_d3',
               'ringnorm_d5',
               'ringnorm_d7',
               'threenorm_d2',
               'threenorm_d3',
               'threenorm_d5',
               'threenorm_d7',
               'twonorm_d2',
               'twonorm_d3',
               'twonorm_d5',
               'twonorm_d7',
               'classification_c2_d2',
               'classification_c2_d3',
               'classification_c2_d5',
               'classification_c2_d7',
               'gaussian_quantiles_c2_d2',
               'gaussian_quantiles_c2_d3',
               'gaussian_quantiles_c2_d5',
               'gaussian_quantiles_c2_d7',
               'blobs_c2_d2',
               'blobs_c2_d3',
               'blobs_c2_d5',
               'blobs_c2_d7']
#total c3 = 20
datasets_c3 = ['cassini',
               '2dnormals_c3',
               'xor_d2',
               'waveform',
               'spirals_c3',
               'dinisurface',
               'helicoid',
               'swissroll',
               'classification_c3_d2',
               'classification_c3_d3',
               'classification_c3_d5',
               'classification_c3_d7',
               'gaussian_quantiles_c3_d2',
               'gaussian_quantiles_c3_d3',
               'gaussian_quantiles_c3_d5',
               'gaussian_quantiles_c3_d7',
               'blobs_c3_d2',
               'blobs_c3_d3',
               'blobs_c3_d5',
               'blobs_c3_d7']
#total c4 = 4
datasets_c4 = ['shapes',
               'smiley',
               'cuboids',
               'hypercube_d2']
#total c5 = 13
datasets_c5 = ['2dnormals_c5', 
               'spirals_c5',
#                'classification_c5_d2', #C{5,7} does not work with d=2
               'classification_c5_d3',
               'classification_c5_d5',
               'classification_c5_d7',
               'gaussian_quantiles_c5_d2',
               'gaussian_quantiles_c5_d3',
               'gaussian_quantiles_c5_d5',
               'gaussian_quantiles_c5_d7',
               'blobs_c5_d2',
               'blobs_c5_d3',
               'blobs_c5_d5',
               'blobs_c5_d7']
#total c7 = 14
datasets_c7 = ['2dnormals_c7',
               'xor_d3',
               'spirals_c7',
#                'classification_c5_d2', #C{5,7} does not work with d=2
               'classification_c7_d3',
               'classification_c7_d5',
               'classification_c7_d7',
               'gaussian_quantiles_c7_d2',
               'gaussian_quantiles_c7_d3',
               'gaussian_quantiles_c7_d5',
               'gaussian_quantiles_c7_d7',
               'blobs_c7_d2',
               'blobs_c7_d3',
               'blobs_c7_d5',
               'blobs_c7_d7']
#total c8 = 1
datasets_c8 = ['hypercube_d3']

time_start = pd.datetime.now()
print(time_start)
base_path = 'E:/extras/datasets'

dataset_names = datasets_d2 + datasets_d3 + datasets_d5 + datasets_d7 + datasets_d21 → 

nTrees = 64
noise_features = [1,2,4,8,16,32,64,128,256,512,1024]
noises_percentual = [5,10,20,40] #% of noisy instances
orientations = ['g', 'in', 'out']
metrics = ['str', 'eigen','katz']
graphEdgesWeightMetric = 'binary'


output_path = 'E:/extras/outputs'
create_output_files(output_path)


for dataset_name in dataset_names:
    Parallel(n_jobs=10)(delayed(ranker)(kfold) for kfold in range(1,11)) 
    print(pd.datetime.now())
print(pd.datetime.now())