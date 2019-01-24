import numpy as np
import matplotlib.pyplot as plt
import agent
from unityagents import UnityEnvironment
from configs import get_dqn_cfg_defaults
import agent_interfaces
plt.ion()
cfgs = get_dqn_cfg_defaults().HYPER_PARAMETER


if __name__ == '__main__':
    # init the unit banana collect env
    env = UnityEnvironment(file_name="./Banana_Env/Banana_Linux/Banana.x86_64")
    # get brain name
    brain_name = env.brain_names[0]
    # get the brain
    brain = env.brains[brain_name]
    # reset the env
    env_info = env.reset(train_mode=True)[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    # create agent based on the config file in module config.agent_dqn_config.py
    agent = getattr(agent, cfgs.AGENT_TYPE)(state_size, action_size, seed=0)
    # create agent-interface based on the config file in module config.agent_dqn_config.py
    scores = getattr(agent_interfaces, cfgs.INTERFACE)(agent, env, brain_name)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
