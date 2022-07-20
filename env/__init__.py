from gym.envs.registration import register

register(
    id='cylinder_robot_pushing_recBox-v0',
    entry_point='env.robot_pushing:cylinder_robot_pushing_recBox',
)
