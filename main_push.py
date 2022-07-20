import gym
import numpy as np
from MPPI import *
import myenv

if __name__ == "__main__":
    env = gym.make('cylinder_robot_pushing_recBox-v0')
    goal_pos = [0.0, 0.0]

    # controller
    horizon = 30
    nr_samples = 30

    controller = MPPI(horizon, nr_samples)
    controller.set_goal(goal_pos[0], goal_pos[1])
    controller.init_matrices(env.rob_state)

    # path showing
    env.init_path_candidates(nr_samples)

    # data collection params
    nr_saved = 0
    collect_data = False

    for i_episode in range(300):
        observation = env.reset()   # self.state = np.concatenate([self.rob_state, self.obj_state], 0)

        for t in range(100):
            env.render()
            # update controller info and get action
            controller.update_real_state(env.rob_state)
            action, mpc_paths = controller.get_action()

            env.show_paths_candidates(mpc_paths)

            print("action: ", action)
            observation, reward, done, info = env.step(action)
            
            # init next action
            controller.init_next_action()

            # after 20 timesteps, we save the trajetory
            if (t+1) == 25 and collect_data == True:
                env.save_trajectory_data(nr_saved)
                nr_saved += 1

            if done:
                print("Episode finished after {} timesteps".format(t+1))

                break
    env.close()