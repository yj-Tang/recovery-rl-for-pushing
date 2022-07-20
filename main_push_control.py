import gym
import numpy as np
from controllers.MPPI_Push import *
import env
from Network import *

if __name__ == "__main__":
    env = gym.make('cylinder_robot_pushing_recBox-v0')
    goal_pos = [0.0, 0.0]

    input_size = 12 # nr of input states nn
    output_size = 6 # nr of output states nn
    horizon = 25 # prediction horizon of controller
    nr_samples = 30 # nr of sampling paths controller

    # model
    model = MLP(input_size, output_size)
    model.load_state_dict(torch.load("./trajectories/saved_nn/torch_model_supervised.pt"))

    # controller
    controller = MPPI_Push(horizon, nr_samples)
    controller.update_model(model)
    controller.set_goal(goal_pos[0], goal_pos[1])
    controller.init_matrices(env.state)

    # path showing
    env.init_path_candidates_both(nr_samples)

    # data collection params
    collect_data = False
    nr_saved = 0

    for i_episode in range(300):
        observation = env.reset()   

        for t in range(100):
            env.render()
            # update controller info and get action
            controller.update_real_state(env.state, env.prev_state)
            controller.update_real_state(env.state, env.prev_state)

            action, mpc_paths = controller.get_action()

            # makes it slow - comment for faster sim
            env.show_paths_candidates_both(mpc_paths)

            print("action: ", action)
            observation, reward, done, info = env.step(action)

            # init next action controller
            controller.init_next_action()

            # after 20 timesteps, we save the trajetory
            if (t+1) == 25 and collect_data == True:
                env.save_trajectory_data(nr_saved)
                nr_saved += 1

            if done:
                print("Episode finished after {} timesteps".format(t+1))

                break
    env.close()