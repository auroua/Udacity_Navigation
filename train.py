import numpy as np
import matplotlib.pyplot as plt
import agent
from unityagents import UnityEnvironment
from configs import get_cfg_defaults
import agent_interfaces
plt.ion()
cfgs = get_cfg_defaults().HYPER_PARAMETER


if __name__ == '__main__':
    env = UnityEnvironment(file_name="./Banana_Env/Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    agent = getattr(agent, cfgs.AGENT_TYPE)(state_size, action_size, seed=0)
    scores = getattr(agent_interfaces, cfgs.INTERFACE)(agent, env, brain_name)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
