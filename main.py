from random_info_maker import make_random_info_list
from comparison_maker import make_comparison_list
    vec_obs_size=1
    act_size=1
    stack_num = 1
    info_list = make_random_info_list(10, vec_obs_size, act_size, stack_num)
    comparison_list = make_comparison_list(10,info_list)
    sess = tf.Session()
    predict_model = PredictModel(sess, vec_obs_size, act_size, stack_num, scope='', layer_num=3)
    sess.run(tf.global_variables_initializer())
    loss = predict_model.update_model(comparison_list)
    pprint(loss)
