import numpy as np
from pprint import pprint

import grid2op
from grid2op.Exceptions import AmbiguousAction

from oracle4grid.core.utils.actions_generator import get_valid_sub_action, get_valid_line_action

class OracleAction:

    def __init__(self, atomic_actions, action_space, debug = False):
        self.debug = debug
        self.action_space = action_space
        self.atomic_actions = atomic_actions
        self.subs, self.lines = self.compute_subs_and_lines()
        #self.grid2op_action_dict = self.get_valid_grid2op_action()

    def compute_subs_and_lines(self):
        subs = set(atomic_action[1] for atomic_action in self.atomic_actions if atomic_action[0]=='sub')
        lines = set(atomic_action[1] for atomic_action in self.atomic_actions if atomic_action[0] == 'line')
        return subs, lines

    def compute_lines(self):
        lines = set()
        for atomic_action in self.atomic_actions:
            if 'line' in self.atomic_actions:
                lines = lines.union(set(self.atomic_actions['sub'].keys()))
        return lines

    def get_subs(self):
        return self.subs

    def get_lines(self):
        return self.lines

    def get_atomic_actions(self):
        return self.atomic_actions

    def get_depth(self):
        return len(self.atomic_actions)

    def print(self):
        print("Action of depth "+str(self.get_depth()))
        print("Acting on "+str(len(self.subs))+" substations and "+str(len(self.lines))+" lines")
        pprint(self.atomic_actions)
        return


    def get_staged_dictionary(self, sub_line = 'sub'):
        dico = {}
        if sub_line == 'sub':
            for sub_id in self.subs:
                dico[int(sub_id)] = [atomic_action[2] for atomic_action in self.atomic_actions if (atomic_action[0]=='sub' and atomic_action[1]==sub_id)]
        elif sub_line == 'line':
            for line_id in self.lines:
                dico[int(line_id)] = [atomic_action[2] for atomic_action in self.atomic_actions if (atomic_action[0]=='line' and atomic_action[1]==line_id)]
        return dico

    def get_valid_grid2op_action(self):
        if len(self.subs) > 0:
            sub_dict = self.get_staged_dictionary(sub_line='sub')
            proper_sub_action = get_valid_sub_action(self.action_space, sub_dict)
        else:
            proper_sub_action = {}
        if len(self.lines) > 0:
            line_dict = self.get_staged_dictionary(sub_line='line')
            proper_line_action = get_valid_line_action(line_dict)
        else:
            proper_line_action = {}

        #     action_dict = {"set_bus": proper_sub_action,
        #                    "set_line_status": proper_line_action,
        #                    }
        action_dict = {**proper_sub_action, **proper_line_action}
        action_grid2op = self.action_space(action_dict)
        if self.debug:
            print(f'State: \n-----\n{action_dict}\n')
            print(action_grid2op)
        return action_grid2op, action_dict