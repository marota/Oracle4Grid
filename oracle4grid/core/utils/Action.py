import numpy as np
from pprint import pprint

import grid2op
from grid2op.Exceptions import AmbiguousAction

from oracle4grid.core.utils.actions_generator import get_valid_sub_action, get_valid_line_action, get_first_key, merge_list_of_dict


class OracleAction:

    def __init__(self, id_, atomic_actions, action_space, init_topo_vect, init_line_status, debug=False):
        self.name = id_
        self.debug = debug
        self.atomic_actions = atomic_actions
        self.subs, self.lines = self.compute_subs_and_lines()
        self.grid2op_action, self.grid2op_action_dict = self.get_valid_grid2op_action(action_space, init_topo_vect, init_line_status)
        self.topo_subids = self.get_topo_subids(action_space)
        # On ne veut pas stocker action_space et init_topo_vect mais juste les utiliser une fois dans la méthode qui formatte l'action en grid2Op

    def __str__(self):
        # TODO: use id of "unitary" OracleAction
        result = str(self.name)
        for sub in self.subs:
            result += '_sub_'+str(sub)
        for line in self.lines:
            result += '_line_'+str(line)
        return result

    def __repr__(self):
        return self.__str__()

    def compute_subs_and_lines(self):
        subs = set(get_first_key(atomic_action['sub'])
                   for atomic_action in self.atomic_actions
                   if get_first_key(atomic_action) == 'sub')
        lines = set(get_first_key(atomic_action['line'])
                    for atomic_action in self.atomic_actions
                    if get_first_key(atomic_action) == 'line')
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
        print("Grid2op resulting topology:")
        pprint(self.grid2op_action_dict)
        return

    def get_valid_grid2op_action(self, action_space, init_topo_vect, init_line_status):
        # Get proper action combination on substations if relevant
        if len(self.subs) > 0:
            sub_actions = [atomic_action['sub']
                           for atomic_action in self.atomic_actions
                           if get_first_key(atomic_action) == 'sub']
            sub_actions_merged = merge_list_of_dict(sub_actions)
            proper_sub_action = get_valid_sub_action(action_space, sub_actions_merged, init_topo_vect)
        else:
            proper_sub_action = {}

        # Get proper action combination on lines if relevant
        if len(self.lines) > 0:
            line_actions = [atomic_action['line']
                            for atomic_action in self.atomic_actions
                            if get_first_key(atomic_action) == 'line']
            line_actions_merged = merge_list_of_dict(line_actions)
            proper_line_action = get_valid_line_action(line_actions_merged, init_line_status)
        else:
            proper_line_action = {}

        #     action_dict = {"set_bus": proper_sub_action,
        #                    "set_line_status": proper_line_action,
        #                    }

        # Get proper full action for grid2op
        action_dict = {**proper_sub_action, **proper_line_action}
        action_grid2op = action_space(action_dict)
        if self.debug:
            print(f'State: \n-----\n{action_dict}\n')
            print(action_grid2op)
        return action_grid2op, action_dict

    def transition_action_to(self, action, init_topo_vect, init_line_status):
        ## Computes action to go from self to action
        action_g2op_1 = self.grid2op_action_dict.copy()
        action_g2op_2 = action.grid2op_action_dict.copy()

        # If no action on bus or line: it is init configuration
        for action_g2op in [action_g2op_1, action_g2op_2]:
            if 'set_bus' not in action_g2op.keys():
                action_g2op['set_bus'] = init_topo_vect.copy()
            if 'set_line_status' not in action_g2op.keys():
                action_g2op['set_line_status'] = init_line_status.copy()

        # Compute difference in bus topologies (action 2 compared to action 1)
        diff_topo = [a2 if a1 != a2 else 0 for a1, a2 in zip(action_g2op_1['set_bus'], action_g2op_2['set_bus'])]
        diff_lines = [a2 if a1 != a2 else 0 for a1, a2 in zip(action_g2op_1['set_line_status'], action_g2op_2['set_line_status'])]

        return {'set_bus': np.array(diff_topo), 'set_line_status':np.array(diff_lines)}

    def modified_subs_to(self, action, init_topo_vect):
        # Computes modified subs to go from action1 to action2 with initial topo
        action_g2op_1 = self.grid2op_action_dict.copy()
        action_g2op_2 = action.grid2op_action_dict.copy()

        # If no action on bus or line: it is init configuration
        for action_g2op in [action_g2op_1, action_g2op_2]:
            if 'set_bus' not in action_g2op.keys():
                action_g2op['set_bus'] = init_topo_vect.copy()

        # Compute modified substation by action 2 compared to action 1
        modified_subs = {sub_id
                         for sub_id, a1, a2 in
                         zip(self.topo_subids, action_g2op_1['set_bus'], action_g2op_2['set_bus'])
                         if a1 != a2}
        return modified_subs

    def modified_lines_to(self, action, init_line_status):
        # Computes modified lines connexions to go from action1 to action2 with initial connexions
        ## Computes action to go from self to action
        action_g2op_1 = self.grid2op_action_dict.copy()
        action_g2op_2 = action.grid2op_action_dict.copy()

        # If no action on bus or line: it is init configuration
        for action_g2op in [action_g2op_1, action_g2op_2]:
            if 'set_line_status' not in action_g2op.keys():
                action_g2op['set_line_status'] = init_line_status.copy()

        # Compute modified lines by action 2 compared to action 1
        modified_lines = {i
                          for i, (a1, a2) in
                          enumerate(zip(action_g2op_1['set_line_status'], action_g2op_2['set_line_status']))
                          if a1 != a2}
        return modified_lines

    def get_topo_subids(self, action_space):
        # Returns topo vector with corresponding sub ids of each element
        n = action_space.dim_topo
        sub_topo = np.zeros(n, dtype=int)

        # Generators
        gens_pos = action_space.gen_pos_topo_vect
        gens_subids = action_space.gen_to_subid
        for i, pos in enumerate(gens_pos):
            sub_topo[pos] = gens_subids[i]

        # Loads
        loads_pos = action_space.load_pos_topo_vect
        loads_subids = action_space.load_to_subid
        for i, pos in enumerate(loads_pos):
            sub_topo[pos] = loads_subids[i]

        # Line origins
        line_or_pos = action_space.line_or_pos_topo_vect
        line_or_subids = action_space.line_or_to_subid
        for i, pos in enumerate(line_or_pos):
            sub_topo[pos] = line_or_subids[i]

        # Line extremities
        line_ex_pos = action_space.line_ex_pos_topo_vect
        line_ex_subids = action_space.line_ex_to_subid
        for i, pos in enumerate(line_ex_pos):
            sub_topo[pos] = int(line_ex_subids[i])
        return sub_topo

