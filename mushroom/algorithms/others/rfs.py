# Authors: Matteo Pirotta <matteo.pirotta@polimi.it>
#          Marcello Restelli <marcello.restelli@polimi.it>
#
# License: BSD 3 clause

"""Recursive feature selection"""

from __future__ import print_function
import numpy as np
from sklearn.utils import check_array
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.feature_selection.base import SelectorMixin
import sklearn
import time

if sklearn.__version__ == '0.17':
    from sklearn.cross_validation import cross_val_score, check_cv
else:
    from sklearn.model_selection import cross_val_score, check_cv


class rfs_node(object):
    def __init__(self, id, fs_index, fs_name=''):
        self.id = id
        self.feature_index = fs_index
        self.feature_name = fs_name
        self.children = []
        self.data = {}

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "N[id: {}, feat: ({}, {}), children: {}, data: {}]".format(self.id, self.feature_index,
                                                                          self.feature_name,
                                                                          self.children, self.data)


class RFS(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    def __init__(self, feature_selector, features_names=None, verbose=0):
        self.feature_selector = feature_selector
        self.features_names = features_names
        self.verbose = verbose

    def fit(self, state, actions, next_states, reward):
        """Fit the RFS model. The input data is a set of transitions
        (state, action, next_state, reward).

        Parameters
        ----------
        state : {array-like, sparse matrix}, shape = [n_samples, n_states]
            The set of states.

        actions : {array-like, sparse matrix}, shape = [n_samples, n_actions]
            The set of actions.

        next_states : {array-like, sparse matrix}, shape = [n_samples, n_states]
            The set of states reached by applying the given action in the given state.

        reward : {array-like, sparse matrix}, shape = [n_samples, n_rewards]
            The set of rewords associate to the transition.
        """
        check_array(state, accept_sparse=True)
        check_array(actions, accept_sparse=True)
        check_array(next_states, accept_sparse=True)
        check_array(reward.reshape(-1, 1), accept_sparse=True)
        return self._fit(state, actions, next_states, reward)

    def _fit(self, states, actions, next_states, reward):
        X = np.column_stack((states, actions))
        # support = np.zeros(X.shape[1], dtype=np.bool)
        support = []
        self.n_features = X.shape[1]

        # start building the tree of dependences
        node = rfs_node(0, -1, 'Reward')
        self.nodes = [node]

        self.index_support_ = self._recursive_step(X, next_states, reward, support, node.id)
        print()
        print('Fit ended')
        print()
        return self

    def _recursive_step(self, X, next_state, Y, curr_support, parent_node_id, Y_idx=None):
        """
        Recursively selects the _implementations that explains the provided target
        (initially Y must be the reward)
        Args:
            X (numpy.array): _implementations. shape = [n_samples, (state_dim + action_dim)]
            next_state (numpy.array): _implementations of the next state [n_samples,  state_dim]
            Y (numpy.array): target to fit (intially reward, than the state)
            curr_support (numpy.array): selected _implementations of X (ie. selected state and action).
                Boolean array of shape [state_dim + action_dim, 1]
            Y_idx (int): index of the target variable

        Returns:
            support (numpy.array): updated support

        """
        if self.verbose > 0:
            print('')
            print('=' * 20)
            if Y_idx is None:
                print('Explaining feature REWARD')
            else:
                print('Explaining feature {}'.format(self.features_names[Y_idx]))
            print('Calling IFS...')

        n_states = next_state.shape[1]
        # n_actions = X.shape[1] - n_states

        fs = clone(self.feature_selector)

        if hasattr(fs, 'set_feature_names'):
            fs.set_feature_names(self.features_names)

        start_t = time.time()
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)

        fs.fit(X, Y)
        end_t = time.time() - start_t
        if self.verbose > 0:
            print('IFS done in {}s'.format(end_t))

        # sa_support = fs.get_support()  # get selected _implementations of X

        # update the tree of dependences
        sa_indexes = fs.get_support(indices=True)
        parent_node = self.nodes[parent_node_id]
        parent_node.data.update({'r2score': fs.scores_})
        parent_node.data.update({'ordered_features': fs.features_per_it_})
        new_node_id = len(self.nodes)
        for k in sa_indexes:
            node = rfs_node(new_node_id, k, self.features_names[k])
            parent_node.children.append(new_node_id)
            self.nodes.append(node)
            new_node_id += 1

        # new_state_support = sa_support[:n_states]  # get only state _implementations
        new_state_indexes = sa_indexes[sa_indexes < n_states]  # get only state _implementations
        # new_state_support[curr_support[:n_states]] = False  # remove state _implementations already selected
        # idxs = np.where(new_state_support)[0]  # get numerical index
        idxs = np.setdiff1d(new_state_indexes, curr_support)  # remove already selected _implementations

        # update support with _implementations already selected
        # new_support + old_support
        # sa_support[curr_support] = True
        sa_indexes = np.union1d(sa_indexes, curr_support).astype(int)

        if self.verbose > 0:
            # print('Selected _implementations {}'.format(self.features_names[sa_support]))
            print('Selected _implementations {}'.format(self.features_names[sa_indexes]))
            print('Feature to explain {}'.format(self.features_names[idxs]))

        for feat_id in idxs:
            v = [self.nodes[el].id for el in parent_node.children if self.nodes[el].feature_index == feat_id]
            assert len(v) == 1
            pnode_id = v[0]
            target = next_state[:, feat_id]
            rfs_s_features = self._recursive_step(X, next_state, target,
                                                  sa_indexes, pnode_id, feat_id)
            # sa_support[rfs_s_features] = True
            sa_indexes = np.union1d(sa_indexes, rfs_s_features)
        # return sa_support
        return sa_indexes

    def _get_support_mask(self):
        """
        The selected _implementations of state and action
        Returns:
            support (numpy.array): the selected _implementations of
                state and action
        """
        # return self.support_
        sup = np.zeros(self.n_features, dtype=np.bool)
        sup[self.index_support_] = True
        return sup

    def export_graphviz(self, filename='rfstree.gv'):
        """
        Export the dependency graph built in the fir phase
        as a graphviz document. It returns an object g representing the
        graph (e.g., you can visualize it by g.view())
        Args:
            filename (str): output file

        Returns:
            g (graphviz.Digraph): an object representing the graph

        """
        def apply_styles(graph, styles):
            graph.graph_attr.update(
                ('graph' in styles and styles['graph']) or {}
            )
            graph.node_attr.update(
                ('nodes' in styles and styles['nodes']) or {}
            )
            graph.edge_attr.update(
                ('edges' in styles and styles['edges']) or {}
            )
            return graph

        if not hasattr(self, 'nodes'):
            raise ValueError('Model must be trained.')

        from graphviz import Digraph

        g = Digraph('G', filename=filename)
        g.body.extend(['rankdir=BT'])
        g.attr('node', shape='circle')
        # BFS
        S = set()
        Q = [0]
        g.node('0', label='{}\nr2={:.4f}'.format(self.nodes[0].feature_name, self.nodes[0].data['r2score'][-1]))
        while len(Q) > 0:
            current_id = Q[0]
            current = self.nodes[current_id]
            Q = [Q[i] for i in range(1, len(Q))]

            # prepare visualization data
            keys = {}
            if 'r2score' in current.data.keys():
                diff_scores = np.ediff1d(current.data['r2score'], to_begin=current.data['r2score'][0])
                for cnt, el in enumerate(current.data['ordered_features']):
                    keys[el] = cnt
            else:
                diff_scores = None

            for node_id in current.children:
                if node_id not in S:
                    lfn = self.nodes[node_id].feature_name
                    if current.feature_name == self.nodes[node_id].feature_name:
                        # make self loop if parent feature is equal to the current one
                        g.edge(str(current_id), str(current_id),
                               label='r2={:.4f}'.format(diff_scores[keys[lfn]]) if diff_scores is not None else '')
                    else:
                        if 'r2score' in self.nodes[node_id].data.keys():
                            lbl = '{}\nr2={:.4f}'.format(lfn, self.nodes[node_id].data['r2score'][-1])
                        else:
                            lbl = '{}'.format(lfn)
                        g.node(str(node_id), label=lbl)
                        g.edge(str(node_id), str(current.id),
                               label='r2={:.4f}'.format(diff_scores[keys[lfn]]) if diff_scores is not None else '')
                    S.add(node_id)
                    Q.append(node_id)

        styles = {
            # 'graph': {
            #     'label': 'A Fancy Graph',
            #     'fontsize': '16',
            #     'fontcolor': 'black',
            #     'bgcolor': 'white',
            #     'rankdir': 'BT',
            # },
            # 'nodes': {
            #     'fontname': 'Helvetica',
            #     'shape': 'hexagon',
            #     'fontcolor': 'black',
            #     'color': 'black',
            #     'style': 'filled',
            #     'fillcolor': 'white',
            # },
            'edges': {
                # 'style': 'solid',
                # 'color': 'black',
                'arrowhead': 'open',
                # 'fontname': 'Courier',
                'fontsize': '12',
                'fontcolor': 'black',
            }
        }
        g = apply_styles(g, styles)
        # g.view()
        return g