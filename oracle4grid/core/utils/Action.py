import numpy as np
from pprint import pprint

import grid2op
from grid2op.Exceptions import AmbiguousAction

from oracle4grid.core.utils.actions_generator import get_first_key, format_sub_action_dict_for_g2op, format_line_action_dict_for_g2op


class OracleAction:

    def __init__(self, id_, atomic_actions_names, atomic_actions, action_space, debug=False):
        self.name = id_
        self.repr = self.get_action_representation(atomic_actions_names)
        self.debug = debug
        self.atomic_actions = atomic_actions
        self.subs, self.lines = self.compute_subs_and_lines(atomic_actions_names)
        self.grid2op_action = self.get_valid_grid2op_action(action_space)
        # On ne veut pas stocker action_space mais juste l'utiliser une fois dans la méthode qui formatte l'action en grid2Op

    def __str__(self):
        return self.repr

    def __repr__(self):
        return self.repr

    def get_action_representation(self, atomic_action_names):
        return "_".join(atomic_action_names)

    def get_atomic_action_by_repr(self, repr):
        atomic_actions_names = self.repr.split("_")
        try:
            i = atomic_actions_names.index(repr)
            return self.atomic_actions[i]
        except:
            return None

    def compute_subs_and_lines(self,atomic_actions_names):
        #subs = set(get_first_key(atomic_action['sub'])
        #           for atomic_action in self.atomic_actions
        #           if get_first_key(atomic_action) == 'sub')
        subs = {get_first_key(self.atomic_actions[i]['sub']): atomic_actions_names[i]
                for i in range(len(self.atomic_actions))
                if get_first_key(self.atomic_actions[i]) == 'sub'}
       #lines = set(get_first_key(atomic_action['line'])
       #            for atomic_action in self.atomic_actions
       #            if get_first_key(atomic_action) == 'line')
        lines = {get_first_key(self.atomic_actions[i]['line']): atomic_actions_names[i]
                for i in range(len(self.atomic_actions))
                if get_first_key(self.atomic_actions[i]) == 'line'}
        return subs, lines

    def get_subs(self):
        return self.subs

    def get_lines(self):
        return self.lines

    def get_atomic_actions(self):
        return self.atomic_actions  # The list of atomic actions that define this combination

    def get_depth(self):
        return len(self.atomic_actions)  # Number of atomic actions combined

    def print(self):
        print("Action of depth " + str(self.get_depth()))
        print("Acting on " + str(len(self.subs)) + " substations and " + str(len(self.lines)) + " lines")
        pprint(self.atomic_actions)
        print("Grid2op resulting impact:")
        pprint(self.grid2op_action.impact_on_objects())
        return

    def get_valid_grid2op_action(self, action_space):
        action_grid2op = action_space()
        for atomic_action in self.atomic_actions:
            if get_first_key(atomic_action) == 'sub':
                formated_atomic_action = format_sub_action_dict_for_g2op(atomic_action['sub'], action_space)
            elif get_first_key(atomic_action) == 'line':
                formated_atomic_action = format_line_action_dict_for_g2op(atomic_action['line'])
            else: # do nothing
                formated_atomic_action = {}
            action_grid2op += action_space(formated_atomic_action)
        return action_grid2op

    def transition_action_to(self, action):
        action_g2op_1 = self.grid2op_action
        action_g2op_2 = action.grid2op_action
        return action_g2op_1 + action_g2op_2

    def number_of_modified_subs_to(self, action):
        subs_1=set(self.subs.keys())
        subs_2=set(action.subs.keys())
        different_Subs=subs_1.symmetric_difference(subs_2)
        same_subs=subs_1.intersection(subs_2)
        same_subs_different_topo=[sub for sub in same_subs if self.subs[sub]!=action.subs[sub]]

        return len(different_Subs)+len(same_subs_different_topo)

    def number_of_modified_lines_to(self, action):
        lines_1=set(self.lines.keys())
        lines_2=set(action.lines.keys())
        different_lines=lines_1.symmetric_difference(lines_2)
        same_lines=lines_1.intersection(lines_2)
        same_lines_different_status=[line for line in same_lines if self.lines[line]!=action.lines[line]]

        return len(different_lines)+len(same_lines_different_status)
