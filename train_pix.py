import numpy as np
import matplotlib.pyplot as plt
import agent
from unityagents import UnityEnvironment
from configs import get_dqn_pix_cfg_defaults as cfgs_func
import agent_interfaces
plt.ion()
cfgs = cfgs_func().HYPER_PARAMETER


if __name__ == '__main__':
    env = UnityEnvironment(file_name="./Banana_Env/Banana_Linux_Pixels/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # examine the state space
    state = env_info.visual_observations[0]
    agent = getattr(agent, cfgs.AGENT_TYPE)(state.shape, action_size, seed=0)
    scores = getattr(agent_interfaces, cfgs.INTERFACE)(agent, env, brain_name)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
