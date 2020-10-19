import numpy as np
import grid2op
from grid2op.Exceptions import AmbiguousAction

class Action:

    def __init__(self, atomic_actions, debug = False):
        self.debug = debug
        self.atomic_actions = atomic_actions
        self.subs = self.compute_subs()
        self.lines = self.compute_lines()
        self.LINE_ON_SUB_ERR = "Line id {} is not connected to sub id {}"
        self.GEN_ON_SUB_ERR = "Generator id {} is not connected to sub id {}"
        self.LOAD_ON_SUB_ERR = "Load id {} is not connected to sub id {}"
        self.grid2op_action_dict = self.get_valid_grid2op_action(atomic_actions)

    def compute_subs(self):
        subs = {int(atomic_action[0].split('_')[1]) for atomic_action in self.atomic_actions if 'sub_' in atomic_action[0]}
        return subs

    def compute_lines(self):
        # TODO
        return

    def get_subs(self):
        return self.subs

    def get_lines(self):
        return self.lines

    def get_atomic_actions(self):
        return self.atomic_actions

    def get_depth(self):
        return len(self.atomic_actions)

    def print(self):
        # return string representing action
        return

    def to_grid_to_op(self):
        # return to grid2op format
        return


    def get_valid_grid2op_action(self):
        # TODO: pas env grid2op, extraire subdict et linedict de self.atomic_actions


        assert isinstance(sub_dict, dict)
        assert isinstance(line_dict, dict)

        env.reset()
        proper_sub_action = self.get_valid_sub_action(sub_dict)
        proper_line_action = self.get_valid_line_action(line_dict)

        #     action_dict = {"set_bus": proper_sub_action,
        #                    "set_line_status": proper_line_action,
        #                    }
        action_dict = {**proper_sub_action, **proper_line_action}
        #action_grid2op = env.action_space(action_dict)
        if self.debug:
            print(f'State: \n-----\n{action_dict}\n')
            print(action_grid2op)
        return action_grid2op, action_dict

    def get_valid_sub_action(self, dict_):
        action_space = self.env.action_space
        set_bus_vect = np.zeros(action_space.dim_topo, dtype=np.int32)

        assert isinstance(dict_, dict)
        # assert "sub_elems" in dict_
        # assert isinstance(dict_["sub_elems"], dict)
        # ddict_ = dict_["sub_elems"]
        ddict_ = dict_

        # Update provided subs
        for sub_id, sub_elems_dict in ddict_.items():
            sub_start_pos = np.sum(action_space.sub_info[:sub_id])
            sub_end_pos = sub_start_pos + action_space.sub_info[sub_id]
            sub_range_pos = np.arange(sub_start_pos, sub_end_pos).astype(np.int32)

            # Update provided lines buses on sub
            if "lines_id_bus" in sub_elems_dict:
                for line_id, bus_id in sub_elems_dict["lines_id_bus"]:
                    # Get line or and ex topo pos
                    line_pos_or = action_space.line_or_pos_topo_vect[line_id]
                    line_pos_ex = action_space.line_ex_pos_topo_vect[line_id]
                    line_pos = -1
                    # Is line or on sub ?
                    if line_pos_or in sub_range_pos:
                        line_pos = line_pos_or
                    # Is line ex on sub ?
                    if line_pos_ex in sub_range_pos:
                        line_pos = line_pos_ex

                    # Line not on sub : Error
                    if line_pos == -1:
                        err_msg = self.LINE_ON_SUB_ERR.format(line_id, sub_id)
                        raise AmbiguousAction(err_msg)
                    else:  # Set line bus on sub
                        set_bus_vect[line_pos] = bus_id

            # Set provided gens buses on sub
            if "gens_id_bus" in sub_elems_dict:
                for gen_id, bus_id in sub_elems_dict["gens_id_bus"]:
                    # Get gen pos in topo
                    gen_pos = action_space.gen_pos_topo_vect[gen_id]
                    # Gen not on sub: Error
                    if gen_pos not in sub_range_pos:
                        err_msg = self.GEN_ON_SUB_ERR.format(gen_id, sub_id)
                        raise AmbiguousAction(err_msg)
                    else:  # Set gen bus on sub
                        set_bus_vect[gen_pos] = bus_id

            # Set provided loads buses on sub
            if "loads_id_bus" in sub_elems_dict:
                for load_id, bus_id in sub_elems_dict["loads_id_bus"]:
                    # Get load pos in topo
                    load_pos = action_space.load_pos_topo_vect[load_id]
                    # Load not on sub: Error
                    if load_pos not in sub_range_pos:
                        err_msg = self.LOAD_ON_SUB_ERR.format(load_id, sub_id)
                        raise AmbiguousAction(err_msg)
                    else:  # Set load bus on sub
                        set_bus_vect[load_pos] = bus_id

        return {"set_bus": set_bus_vect}

    def get_valid_line_action(self, line_dict):
        return {"set_line_status": [(k, v['set_line']) for k, v in line_dict.items()]}