#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : morphology.py
#   Author       : Yufeng Liu
#   Date         : 2021-07-16
#   Description  : The script is responsible for morphological and 
#                  topological feature calculation, including: 
#                   - morphological: 
#                   - topological:
#                  The features: #stems, #tips, #total_length, #depth and #branches
#                  are the same as that in Vaa3D, tested on two randomly selected
#                  swc file. The #bifurcation of our package does not include 
#                  the soma point, while Vaa3D does.
#
#================================================================

import copy
import numpy as np
import sys
from swc_handler import get_child_dict, get_index_dict, find_soma_node, find_soma_index, NEURITE_TYPES

sys.setrecursionlimit(100000)


class TreeInitializeError(RuntimeError):
    def __init__(self, args):
        self.args = args


class AbstractTree(object):
    def __init__(self, tree, p_soma=-1):
        self.p_soma = p_soma

        self.tree = tree    # swc tree file, type as list
        if len(self.tree) == 0:
            raise TreeInitializeError("The tree contains no nodes!")

        self.child_dict = get_child_dict(tree)
        self.index_dict = get_index_dict(tree)
        self.pos_dict = self.get_pos_dict()
        self.idx_soma = find_soma_node(tree, p_soma=self.p_soma)    # node index
        self.index_soma = find_soma_index(tree, p_soma)

    def get_nodes_by_types(self, neurite_type):
        nodes = []
        ntypes = NEURITE_TYPES[neurite_type]
        for node in self.tree:
            type_ = node[1]
            if type_ in ntypes:
                nodes.append(node[0])
        return set(nodes)

    def get_pos_dict(self):
        pos_dict = {}
        for i, leaf in enumerate(self.tree):
            pos_dict[leaf[0]] = leaf
        return pos_dict

    def get_volume_size(self):
        coords = [leaf[2:5] for leaf in self.tree]
        coords = np.array(coords)
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = cmax - cmin
        volume = span.prod()
        return span, volume

    def calc_node_distances(self, spacing=(1, 1, 4)):
        """
        Distance distribution for connecting nodes
        """
        coords1 = []
        coords2 = []
        for idx in self.child_dict:
            if idx == self.p_soma: continue
            coord1 = self.pos_dict[idx][2:5]
            for c_idx in self.child_dict[idx]:
                coord2 = self.pos_dict[c_idx][2:5]
                coords1.append(coord1)
                coords2.append(coord2)
        coords1 = np.array(coords1)
        coords2 = np.array(coords2)
        shift = (coords2 - coords1) * spacing
        dists = np.linalg.norm(shift, axis=1)
        stats = dists.mean(), dists.std(), dists.max(), dists.min()
        return stats

    def get_distances_to_soma(self, spacing=(1, 1, 1)):
        """
        euclidean distance to soma for all nodes
        """
        c_soma = np.array(self.pos_dict[self.idx_soma][2:5])
        coords = np.array([node[2:5] for node in self.tree])
        diff = (coords - c_soma) * spacing
        dists = np.linalg.norm(diff, axis=1)
        return dists

    def get_critical_points(self):
        if len(self.tree) == 0:
            self.stems = set([])
            self.tips = set([])
            self.unifurcation = set([])
            self.bifurcation = set([])
            self.multifurcation = set([])
            return 

        # stems
        self.stems = set(self.child_dict[self.idx_soma])

        # terminal points
        all_nodes_indices = set([leaf[0] for leaf in self.tree])
        has_child_indices = set(list(self.child_dict.keys()))
        self.tips = all_nodes_indices - has_child_indices

        # and bifurcate points
        self.unifurcation = []   # with only one child
        self.bifurcation = []   # 2 childs
        self.multifurcation = [] # > 2 childs
        for idx, childs in self.child_dict.items():
            if idx == self.p_soma:
                continue    # virtual point for soma parent
            if idx == self.idx_soma:
                continue
            if len(childs) == 1:
                self.unifurcation.append(idx)
            elif len(childs) == 2:
                self.bifurcation.append(idx)
            elif len(childs) > 2:
                self.multifurcation.append(idx)
        self.unifurcation = set(self.unifurcation)
        self.bifurcation = set(self.bifurcation)
        self.multifurcation = set(self.multifurcation)

    def get_all_paths(self):
        """
        Find out all paths from tip to soma
        """
        if not hasattr(self, 'tips'):
            self.get_critical_points()

        paths = {}
        for tip in self.tips:
            path = [tip]
            leaf = self.pos_dict[tip]
            while leaf[6] in self.pos_dict:
                pid = leaf[6]
                path.append(pid)
                leaf = self.pos_dict[pid]
            paths[tip] = path
        return paths
        

    def calc_frag_lengths(self):
        """
        calculate all fragment lengths, that is the length between two successive points
        """
        # in parallel mode
        p_coords = []
        for leaf in self.tree:
            if leaf[-1] != self.p_soma:
                p_coords.append(self.pos_dict[leaf[-1]][2:5])
            else:
                p_coords.append(self.pos_dict[self.idx_soma][2:5])
        #counted_leaf = [leaf for leaf in self.tree if leaf[-1] != self.p_soma and leaf[-1] in self.pos_dict]
        #coords = np.array([leaf[2:5] for leaf in counted_leaf])
        #p_coords = np.array([self.pos_dict[leaf[-1]][2:5] for leaf in counted_leaf])
        coords = np.array([leaf[2:5] for leaf in self.tree])
        indices = [leaf[0] for leaf in self.tree]
        vectors = coords - p_coords
        lengths = np.linalg.norm(vectors, axis=1)

        lengths_dict = {}
        for idx, length in zip(indices, lengths):
            lengths_dict[idx] = length
        return lengths, lengths_dict

    def calc_total_length(self):
        seg_lengths, lengths_dict = self.calc_frag_lengths()
        total_length = seg_lengths.sum()
        return total_length


class Morphology(AbstractTree):
    def __init__(self, tree, p_soma=-1):
        super(Morphology, self).__init__(tree, p_soma=p_soma)
        self.get_critical_points()

    def get_path_idx_dict(self):
        """
        Find path from each node to soma. DFS search
        """

        def find_path_dfs(idx, path_dict, pos_dict, child_dict):
            pidx = pos_dict[idx][6]
            
            if pidx in path_dict:
                path_dict[idx] = path_dict[pidx] + [pidx]

            if idx not in child_dict:
                return
            else:
                for cidx in child_dict[idx]:
                    find_path_dfs(cidx, path_dict, pos_dict, child_dict)

        path_dict = {}
        path_dict[self.idx_soma] = []
        print(len(self.tree))
        find_path_dfs(self.idx_soma, path_dict, self.pos_dict, self.child_dict)
        return path_dict

    def get_path_len_dict(self, path_dict, frag_lengths):
        """
        estimate the path length for each node
        :params path_dict:  node-to-soma dict or tip-to-soma dict, values is all parent index
        :params frag_lengths[list/array]:   fragment lengths
        """
        plen_dict = {}
        for idx, pidxs in path_dict.items():
            pindex = [self.index_dict[pidx] for pidx in pidxs]
            plen = frag_lengths[pindex].sum()
            plen_dict[idx] = plen
        return plen_dict

    def calc_seg_path_lengths(self, seg_dict, frag_lengths_dict):
        """
        segmental path length
        output: dictionary of segmental index -> seg_length. the segmental index are
                the end point of a segment
        """
        path_dists = {}
        path_dists[self.idx_soma] = 0 
        for seg_id, seg_nodes in seg_dict.items():
            path_dists[seg_id] = frag_lengths_dict[seg_id]
            for n_id in seg_nodes:
                path_dists[seg_id] += frag_lengths_dict[n_id]
        return path_dists

    def prune_by_seg_length(self, seg_dict, seglen_dict, seg_length_thresh):
        remaining_tips = [tip for tip in self.tips]
        del_tips = []
        while len(remaining_tips) > 0:
            del_tips_current = []
            del_seg_end_current = []
            for tip in remaining_tips:
                if self.pos_dict[tip][6] == self.idx_soma:
                    continue
                if seglen_dict[tip] < seg_length_thresh:
                    # delete current tip
                    del_tips_current.append(tip)
                    if len(seg_dict[tip]) > 0:
                        del_seg_end_current.append(seg_dict[tip][6])
                    else:
                        del_seg_end_current.append(tip)

            del_tips.extend(del_tips_current)

            remaining_tips = []
            # update the tips
            del_seg_end_current_set = set(del_seg_end_current)
            for i, tip in enumerate(del_tips_current):
                pidx_topo = self.pos_dict[del_seg_end_current[i]][6]
                childs_p = self.child_dict[pidx_topo]
                all_removed = True
                for child in childs_p:
                    if child not in del_seg_end_current_set:
                        all_removed = False
                        break
                if all_removed:
                    remaining_tips.append(pidx_topo)

        del_tips = set(del_tips)
        pruned_tree = []
        for seg_id, seg_nodes in seg_dict.items():
            if seg_id not in del_tips:
                pruned_tree.append(self.pos_dict[seg_id])
                for node_id in seg_nodes:
                    #if node_id == self.idx_soma:
                    #    print(node_id, seg_id)
                    pruned_tree.append(self.pos_dict[node_id])

        return pruned_tree

    def convert_to_topology_tree(self):
        """
        The original tree contains unifurcation, which should be merged
        """

        def update_node(old_node, new_par_id):
            tmp_node = (*old_node[:6], new_par_id, *old_node[7:])
            return tmp_node

        seg_dict = {}
        new_tree = []
        for tip in self.tips:
            seg_dict[tip] = []  # intialize current seg
            cur_node_id = tip
            seg_start_id = tip
            while True:
                par_node_id = self.pos_dict[cur_node_id][6]    # parent node
                if par_node_id == -1:
                    break
                if par_node_id in self.unifurcation:
                    seg_dict[seg_start_id].append(par_node_id)
                else:
                    new_tree.append(update_node(self.pos_dict[seg_start_id], par_node_id))
                    if par_node_id in seg_dict:
                        break
                    else:
                        seg_dict[par_node_id] = []

                    seg_start_id = par_node_id
                
                cur_node_id = par_node_id
        # put the root/soma node
        new_tree.append(self.pos_dict[self.idx_soma])

        print(f'{len(new_tree)} #nodes left after merging of the original {len(self.tree)} # nodes')
        #print(f'{len(self.tips)}, {len(self.bifurcation)}, {len(self.multifurcation)}')
        return new_tree, seg_dict


class Topology(AbstractTree):
    def __init__(self, tree, p_soma=-1):
        super(Topology, self).__init__(tree, p_soma=p_soma)
        self.get_critical_points()
        self.calc_order_dict()

    def calc_order_dict(self):
        # calculate the order of each node as well as largest node through DFS
        # DFS function
        def traverse_dfs(idx, child_dict, order_dict):
            if idx not in child_dict:
                return 

            for child_idx in child_dict[idx]:
                order_dict[child_idx] = order_dict[idx] + 1
                traverse_dfs(child_idx, child_dict, order_dict)

        # Firstly, for topology analysis, we must firstly merge unifurcation nodes
        

        order_dict = {}
        order_dict[self.idx_soma] = 0
        traverse_dfs(self.idx_soma, self.child_dict, order_dict)
        self.order_dict = order_dict

    def get_num_branches(self):
        return len(self.tree) - 1

    def get_topo_width(self):
        """
        Reference to paper: 
            'Modelling brain-wide neuronal morphology via rooted Cayley trees'
        """
        if not hasattr(self, 'order_dict'):
            self.calc_order_dict()

        order_freq_dict = {}
        multifurcation = self.multifurcation | self.bifurcation
        for idx, order in self.order_dict.items():
            if order not in order_freq_dict:
                order_freq_dict[order] = 0

            if idx == self.p_soma:
                order_freq_dict[order] = 1
                continue
            elif idx in multifurcation:
                order_freq_dict[order] += 1
        self.order_freq_dict = order_freq_dict
        self.topo_width = max(order_freq_dict.values())

        return self.topo_width

    def get_topo_depth(self):
        if not hasattr(self, 'order_dict'):
            self.calc_order_dict()
        self.topo_depth = max(self.order_dict.values())
        return self.topo_depth

    def get_topo_segs(self, seg_dict, seglen_dict):
        """
        Convert seg_dict into topology seg, similar to topo_seg in APP2
        Algorithm description
        """
        # dfs search
        def temp(topo_segs, seg_dict, child_dict, idx):
            if idx not in child_dict:
                # tip
                pass

        paths = self.get_all_paths()    # tip to soma
        topo_dists = {}
        topo_leafs = {}
        for node in self.tree:
            topo_dists[node[0]] = 0
            topo_leafs[node[0]] = 0

        for tip, path in paths.items():
            topo_leafs[tip] = tip

            for cnode, pnode in zip(path[:-1], path[1:]):
                tmp_dist = seglen_dict[cnode] + topo_dists[cnode]
                if tmp_dist >= topo_dists[pnode]:
                    topo_dists[pnode] = tmp_dist
                    topo_leafs[pnode] = topo_leafs[cnode]
        #
        topo_segs = {}
        for pid, lid in topo_leafs.items():
            topo_segs[lid] = (topo_leafs[pid], topo_dists[pid])

        return topo_segs
        

if __name__ == '__main__':
    from swc_handler import parse_swc, write_swc

    swcfile = '/media/lyf/storage/seu_mouse/swc/xy1z1/17109_6201_x4328_y6753.swc'
    tree = parse_swc(swcfile)
    morph = Morphology(tree, p_soma=-1)
    new_tree, _ = morph.convert_to_topology_tree()
    
    topo = Topology(new_tree, p_soma=-1)
    #import ipdb; ipdb.set_trace()
    topo.get_topo_width()
    topo.get_topo_depth()
    print(f'''Topology features:
            #stems: {len(morph.stems)}, 
            #tips: {len(morph.tips)}, 
            #unifurcation: {len(morph.unifurcation)}, 
            #bifurcation: {len(morph.bifurcation)},
            #multifurcation: {len(morph.multifurcation)},
            #total_length: {morph.calc_total_length():.2f},
            #dx,dy,dz and volume size: {morph.get_volume_size()}
            #stats for connecting nodes distances: {morph.calc_node_distances()}

            #topo_depth: {topo.topo_depth},
            #topo_width: {topo.topo_width},
            #branches: {topo.get_num_branches()}
            ''')


