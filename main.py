from random_info_maker import make_random_info_list
from comparison_maker import make_comparison_list
    vec_obs_size=1
    act_size=1
    info_list = random_info_list_make(10, vec_obs_size, act_size)
    info_list = make_random_info_list(10, vec_obs_size, act_size)
    comparison_list = make_comparison_list(10,info_list)
