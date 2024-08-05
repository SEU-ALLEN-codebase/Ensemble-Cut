#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : swc_checker.py
#   Author       : Yufeng Liu
#   Date         : 2022-08-04
#   Description  : The script tries to check the common errors of swc file
#
#================================================================

from itertools import groupby
from collections import Counter

from swc_handler import parse_swc
from morph_topo.morphology import Morphology

class AbstractErrorChecker(object):
    def __init__(self, debug=False):
        self.debug = debug

    def __call__(self):
        pass

class MultiSomaChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(MultiSomaChecker, self).__init__(debug)

    def __call__(self, morph):
        num_soma = 0
        for node in morph.tree:
            if node[6] == -1:
                num_soma += 1
        if num_soma > 1:
            if self.debug:
                print(f'Warning: the swc contains {num_soma} somata')
            return False
        else:
            return True

class NoSomaChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(NoSomaChecker, self).__init__(debug)

    def __call__(self, morph):
        num_soma = 0
        for node in morph.tree:
            if node[6] == -1:
                num_soma += 1
        if num_soma == 0:
            if self.debug:
                print(f'Warning: the swc has no soma!')
            return False
        else:
            return True

class ParentZeroIndexChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(ParentZeroIndexChecker, self).__init__(debug)

    def __call__(self, morph):
        for node in morph.tree:
            if node[6] == 0:
                if self.debug:
                    print('Warning: the swc has node with parent index 0!')
                return False
        return True

class MultifurcationChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(MultifurcationChecker, self).__init__(debug)

    def __call__(self, morph):
        if not hasattr(morph, 'multifurcation'):
            morph.get_critical_points()
        no_multifur = len(morph.multifurcation) == 0
        if self.debug and not no_multifur:
            print('Warning: the swc has multifurcation!')

        return no_multifur

class TypeErrorChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(TypeErrorChecker, self).__init__(debug)

    def __call__(self, morph):
        paths = morph.get_all_paths()
        for path in paths.values():
            # get all types, except for the soma
            types = [morph.pos_dict[node][1] for node in path[:-1]]
            types_set = set(types)
            if len(types_set) == 1:
                continue
            else:
                types_group = groupby(types)
                num_switch = len(list(types_group))
                if num_switch > 2:
                    if self.debug:
                        print('Warning: the swc has possible wrong neurite types!')
                        for t, p in zip(types, path[:-1]):
                            print(f'({t}, {p})', end=" ")
                        print('\n')
                    return False
        return True        

class LoopChecker(AbstractErrorChecker):
    def __init__(self, debug):
        super(LoopChecker, self).__init__(debug)

    def __call__(self, morph):
        coords = [node[2:5] for node in morph.tree]
        coords_set = set(coords)
        if len(coords) != len(coords_set):
            if self.debug:
                # for debug only
                print('Warning: duplicated nodes error! Duplicate coordinates are: ')
                counter = Counter(coords)
                for c, cc in counter.items():
                    if cc > 1:
                        print(c, cc)
            return False
        return True

class SWCChecker(object):
    """
    Check the common errors of swc file
    """
    
    ERROR_TYPES = {
        'MultiSoma': 0,
        'NoSoma': 1,
        'ParentZeroIndex': 2,
        'Multifurcation': 3,
        'TypeError': 4,
        'Loop': 5,  # detect nodes with identical coordinates
    }

    def __init__(self, error_types=(), debug=False):
        if not error_types:
            error_types = self.ERROR_TYPES
        
        self.checkers = []
        gvs = globals()
        for error_type in error_types:
            check_name = error_type + 'Checker'
            checker = gvs[check_name](debug=debug)
            self.checkers.append(checker)
        
    def run(self, swcfile):
        if type(swcfile) is str:
            # load swc
            tree = parse_swc(swcfile)
        elif type(swcfile) is list:
            tree = swcfile
        morph = Morphology(tree)
        errors = []
        for checker in self.checkers:
            err = checker(morph)
            errors.append(err)
        return errors
        

if __name__ == '__main__':
    swcfile = '/home/lyf/test.swc'
    #error_types = ('MultiSoma', )
    
    swc_checker = SWCChecker(debug=True)
    print(swc_checker.run(swcfile))

