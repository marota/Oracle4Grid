from grid2op.Agent import BaseAgent
from grid2op.Exceptions import AmbiguousAction
import numpy as np

class OracleAgent(BaseAgent):
    """
    OracleAgent which plays the optimal simulated path in the action transition graph in a scenario
    """
    action_path = []

    def __init__(self,
                 action_space,
                 action_path,
                 oracle_action_path=None,
                 init_topo_vect = None,
                 init_line_status = None,
                 **kwargs):
        BaseAgent.__init__(self, action_space)
        self.action_path = action_path.copy()
        #self.actions_left = self.action_path.copy()
        if(oracle_action_path):
            self.oracle_action_path=oracle_action_path.copy()
        else:
            self.oracle_action_path=None
        # Initialize memory
        self.previous_action = None
        self.current_action = None
        self.init_topo_vect = init_topo_vect
        self.init_line_status = init_line_status
        self.previous_was_legal = True

    def act(self, observation, reward, done):
        action = self.action_path.pop(0)

        # check line reco in addition if some were deconnected not on purpose
        # check if previous atomic_actions has to be canceled thanks to memory
        self.update_memory()
        action_line_reco=self.check_reconnect_line(observation)
        cancelling_action = self.compare_with_previous()
        self.previous_was_legal = True
        if(action_line_reco!=self.action_space({})):
            action_combined=action_line_reco
            action_combined+=action
            if (cancelling_action!=self.action_space({})):
                action_combined += cancelling_action
            if(self.action_space._is_legal(action_combined,observation._obs_env)):
                return action_combined
            else:
                self.previous_was_legal = False

        if (cancelling_action != self.action_space({})):
            action_combined = cancelling_action
            action_combined += action
            if (self.action_space._is_legal(action_combined, observation._obs_env)):
                return action_combined
            else:
                self.previous_was_legal = False
        return action

    def update_memory(self):
        if self.oracle_action_path is not None:
            if self.previous_was_legal:
                self.previous_action = self.current_action
            self.current_action = self.oracle_action_path[0]

    def check_reconnect_line(self,observation):

        res = {}  # add the do nothing
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)

        # don't reconnect lines on which we played on purpose
        if(self.oracle_action_path is not None):
            oracle_action=self.oracle_action_path.pop(0)
            lines_oracle_action=oracle_action.lines
            if(len(lines_oracle_action)>=1):
                for l in lines_oracle_action:
                    can_be_reco[l]=False

        #reconnect useful lines that were in maintenance or attacked
        if np.any(can_be_reco):
            res = {"set_line_status": [(id_, +1) for id_ in np.where(can_be_reco)[0]]}
        return self.action_space(res)

    def compare_with_previous(self):
        if self.previous_action is None or self.init_line_status is None or self.init_topo_vect is None or str(self.previous_action) == 'donothing-0':
            return self.action_space({})
        else:
            # Get previous and current
            previous_atomic_actions_repr = str(self.previous_action).split('_')
            current_atomic_actions_repr = str(self.current_action).split('_')

            # Check if less action
            atomic_actions_to_cancel = []
            for previous_atomic_action_repr in previous_atomic_actions_repr:
                if previous_atomic_action_repr not in current_atomic_actions_repr:
                    if 'line' in previous_atomic_action_repr:
                        atomic_actions_to_cancel.append(self.previous_action.get_atomic_action_by_repr(previous_atomic_action_repr))
                    elif 'sub' in previous_atomic_action_repr:
                        previous_impacted_sub = previous_atomic_action_repr.split('-')[1]
                        current_atomic_actions_repr_same_sub = [repr_ for repr_ in current_atomic_actions_repr
                                                                if 'sub-'+str(previous_impacted_sub) in repr_]
                        if len(current_atomic_actions_repr_same_sub) > 0:
                            # Here we have some possible ambiguous action on same sub
                            current_atomic_actions_same_sub = [self.current_action.get_atomic_action_by_repr(repr_)
                                                               for repr_ in current_atomic_actions_repr_same_sub]
                            previous_atomic_action = self.previous_action.get_atomic_action_by_repr(previous_atomic_action_repr)
                            unambiguous_canceling_action = get_unambiguous_canceling_action(previous_atomic_action,
                                                                                            current_atomic_actions_same_sub,
                                                                                            previous_impacted_sub)
                            atomic_actions_to_cancel.append(unambiguous_canceling_action)
                        else:
                            atomic_actions_to_cancel.append(self.previous_action.get_atomic_action_by_repr(previous_atomic_action_repr))

            # If actions, cancel them
            canceling_g2op_actions = self.action_space({})
            for atomic_action_to_cancel in atomic_actions_to_cancel:
                canceling_g2op_action = get_canceling_action(self.action_space, self.init_line_status, self.init_topo_vect,
                                                             atomic_action_to_cancel)
                canceling_g2op_actions += canceling_g2op_action
            return canceling_g2op_actions

    def reset(self, observation):
        #self.actions_left = self.action_path.copy()
        pass

    def load(self, path):
        # Nothing to load
        pass

    def save(self, path):
        # Nothing to save
        pass

    #@classmethod
    #def gen_next(cls, action_path):
    #    """
    #    This function allows to change the list of action dictionaries that the agent will perform.

    #    See the class level documentation for an example on how to use this.

    #    """
    #    cls.action_path = action_path.copy()
    #    return cls

def get_unambiguous_canceling_action(previous_atomic_action, current_atomic_actions_same_sub, substation_id):
    # 1 - retrieve impact of previous atomic action (line-load-gen)
    previous_atomic_action_to_cancel = {'sub':{int(substation_id):{}}}
    previous_atomic_action_assets = previous_atomic_action['sub'][int(substation_id)]
    for asset in previous_atomic_action_assets.keys():
        list_impacts_previous = previous_atomic_action_assets[asset]
        for current_atomic_action_same_sub in current_atomic_actions_same_sub:
            if asset in current_atomic_action_same_sub['sub'][int(substation_id)].keys():
                list_impact_current = current_atomic_action_same_sub['sub'][int(substation_id)][asset]
                # Filter the impacts who have been found in current
                list_impacts_previous_new = [impact for impact in list_impacts_previous
                                             if impact not in list_impact_current]
            else:
                list_impacts_previous_new = list_impacts_previous
            # If no impact, nothing to add key, else update dict with desired impacts only
            if len(list_impacts_previous_new) > 0:
                previous_atomic_action_to_cancel['sub'][int(substation_id)][asset] = list_impacts_previous_new
    return previous_atomic_action_to_cancel

def get_canceling_action(action_space, init_line_status, init_topo_vect, atomic_action_to_cancel):
    set_bus_vect = np.zeros(action_space.dim_topo, dtype=np.int32)
    LINE_ON_SUB_ERR = "Line id {} is not connected to sub id {}"
    GEN_ON_SUB_ERR = "Generator id {} is not connected to sub id {}"
    LOAD_ON_SUB_ERR = "Load id {} is not connected to sub id {}"

    if 'sub' in list(atomic_action_to_cancel.keys()):
        for sub_id, sub_elems_dict in atomic_action_to_cancel['sub'].items():
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
                        err_msg = LINE_ON_SUB_ERR.format(line_id, sub_id)
                        raise AmbiguousAction(err_msg)
                    else:  # Set line bus on sub
                        set_bus_vect[line_pos] = init_topo_vect[line_pos]

            # Set provided gens buses on sub
            if "gens_id_bus" in sub_elems_dict:
                for gen_id, bus_id in sub_elems_dict["gens_id_bus"]:
                    # Get gen pos in topo
                    gen_pos = action_space.gen_pos_topo_vect[gen_id]
                    # Gen not on sub: Error
                    if gen_pos not in sub_range_pos:
                        err_msg = GEN_ON_SUB_ERR.format(gen_id, sub_id)
                        raise AmbiguousAction(err_msg)
                    else:  # Set gen bus on sub
                        set_bus_vect[gen_pos] = init_topo_vect[gen_pos]

            # Set provided loads buses on sub
            if "loads_id_bus" in sub_elems_dict:
                for load_id, bus_id in sub_elems_dict["loads_id_bus"]:
                    # Get load pos in topo
                    load_pos = action_space.load_pos_topo_vect[load_id]
                    # Load not on sub: Error
                    if load_pos not in sub_range_pos:
                        err_msg = LOAD_ON_SUB_ERR.format(load_id, sub_id)
                        raise AmbiguousAction(err_msg)
                    else:  # Set load bus on sub
                        set_bus_vect[load_pos] = init_topo_vect[load_pos]
        return action_space({"set_bus": set_bus_vect})
    if 'line' in list(atomic_action_to_cancel.keys()):
        action_dict = {"set_line_status": [(line_id, init_line_status[line_id])
                             for line_id, line_elems_dict in atomic_action_to_cancel['line'].items()]}
        return action_space(action_dict)