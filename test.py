import torch
import matplotlib.pyplot as plt
import agent
from unityagents import UnityEnvironment
from configs import get_dqn_cfg_defaults
plt.ion()
cfgs = get_dqn_cfg_defaults().HYPER_PARAMETER


def show_results(env, brain_name, agent):
    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load('./results/checkpoint_dqn_vec.pth',
                                                    map_location=lambda storage, loc: storage))
    with torch.no_grad():
        for i in range(3):
            score = 0
            env_info = env.reset(train_mode=False)[brain_name]
            state = env_info.vector_observations[0]
            while True:
                action = agent.act(state)
                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                state = next_state
                score += reward
                if done:
                    break
            print('episodes %d get score %d' % (i, score))
    env.close()


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
    show_results(env, brain_name, agent)
