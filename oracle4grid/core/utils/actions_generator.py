import string
import numpy as np

from grid2op.Exceptions import AmbiguousAction
from grid2op.PlotGrid import PlotMatplot

LINE_ON_SUB_ERR = "Line id {} is not connected to sub id {}"
GEN_ON_SUB_ERR = "Generator id {} is not connected to sub id {}"
LOAD_ON_SUB_ERR = "Load id {} is not connected to sub id {}"

def get_atomic_actions_names(atomic_actions) :
    # Name of each target configuration
    named_atomic_actions = {}
    for key in atomic_actions:
        dict_sub = {f'{key}_{c}_{i}': [key, c, atomic_actions[key][c][i]] for c in atomic_actions[key] for i,_ in enumerate(atomic_actions[key][c])}
        named_atomic_actions.update(dict_sub)
    return  named_atomic_actions


def init_plot_helper(env, show=False):
    plot_helper = PlotMatplot(env.observation_space,
                              sub_radius=14,
                              load_radius=10,
                              gen_radius=10,
                              width=950,
                              height=600,
                              )
    plot_helper._line_bus_radius = 6
    plot_helper._line_arrow_width = 10
    plot_helper._line_arrow_len = 17

    if show:
        fig = plot_helper.plot_layout()
        fig.show()
    return plot_helper


def plot_action(env, plot_helper, action, line_info=None, load_info=None, gen_info=None):
    obs = env.reset()
    obs, reward, done, info = env.step(env.action_space(action))
    fig_obs = plot_helper.plot_obs(obs, line_info=line_info, load_info=load_info, gen_info=gen_info)
    print('Is ambigous??? -> {}'.format(info['is_ambiguous']))
    print('Exception => {}'.format(info['exception']))
    print()
    return fig_obs


def append_unitary_actions(states, elem, ut):
    elem_type, elem_id = elem.split('_')
    for u_action in ut:
        if elem_type in ['sub','line']:
            if elem_type in states.keys():
                if int(elem_id) not in states[elem_type].keys():
                    states[elem_type] = {**states[elem_type], **{int(elem_id):[u_action[int(elem_id)]]}}
                else:
                    states[elem_type][int(elem_id)].append(u_action[int(elem_id)])
            else:
                states[elem_type] = {int(elem_id):[u_action[int(elem_id)]]}
        else:
            print("ERROR: Elem type should be line or sub")



def get_valid_sub_action(action_space, dict_, init_topo_vect):

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
                    err_msg = LINE_ON_SUB_ERR.format(line_id, sub_id)
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
                    err_msg = GEN_ON_SUB_ERR.format(gen_id, sub_id)
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
                    err_msg = LOAD_ON_SUB_ERR.format(load_id, sub_id)
                    raise AmbiguousAction(err_msg)
                else:  # Set load bus on sub
                    set_bus_vect[load_pos] = bus_id

    return {"set_bus": set_bus_vect}


def get_valid_line_action(line_dict):
    return {"set_line_status": [(k, v['set_line']) for k, v in line_dict.items()]}