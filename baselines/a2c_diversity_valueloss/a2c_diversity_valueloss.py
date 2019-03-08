import time
import functools
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy

# a2c_diversity.utils is the same a2c.utils
# a2c_diversity.runner is the same as a2c.runner
from baselines.a2c_diversity.utils import Scheduler, find_trainable_variables
from baselines.a2c_diversity.runner import Runner

from tensorflow import losses

class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps, log_dir=None,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4, div_coef=0.001,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.get_session()    # create session
        nenvs = env.num_envs            # env includes multiple env
        nbatch = nenvs*nsteps           # batch size


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)     # policy_fn(nbatch, nsteps, sess) = PolicyWithValue(...)
            # step_model.X = tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=dtype, name='Ob')

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        backup_models = []
        for i in range(5):
            with tf.variable_scope('a2c_backup_model_{}'.format(i)):
                # back_up_models is used to restore previous model to calculate diversity
                backup_models.append(policy(nbatch, nsteps, sess, reuse_placeholder=True))

        print("------Print out all variables------")
        variables_names = [v.name for v in tf.trainable_variables()]
        for k in variables_names:
            print ("Variable: ", k)

        print("------Print out all placeholders------")
        for placeholder in tf.contrib.framework.get_placeholders(tf.get_default_graph()):
            print(placeholder)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)       # negative log pi action, -logpi(a|s)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        # Diversity loss version 1 -----> placeholder can't be backpropagated
        # cur_ac = tf.placeholder       # Current action
        # pre_acs = tf.placeholder       # Previous action
        # div_loss = kl_divergence(cur_ac, pre_acs)

        # Diversity loss version 2
        cur_ac = train_model.pd.mean    # current model action; step returns (a, v, state, neglogp)
        pre_acs = [backup_models[i].pd.mean for i in range(5)]    # previous actions

        def average_KL_divergence(cur_ac, pre_acs):
            def kl_divergence(p, q):
                return tf.reduce_sum(p * tf.log(p/q), axis=1)

            kl_divs = []
            for pre_ac in pre_acs:
                kl_divs.append(kl_divergence(cur_ac, pre_ac))    # DKL(π(a|s)||π′(a|s))

            return tf.reduce_sum(kl_divs)  # still need to clip

        div = average_KL_divergence(cur_ac, pre_acs)
        div = div * div_coef
        div = tf.clip_by_value(div, clip_value_min=0, clip_value_max=0.9, name='clipped_diversity')

        loss = pg_loss + vf_loss * vf_coef - tf.stop_gradient(vf_loss * vf_coef) * div  #  - entropy * ent_coef

        # Tensorboard
        tf.summary.scalar('Total loss', loss)
        tf.summary.scalar('Policy loss', pg_loss)
        tf.summary.scalar('Entropy', entropy * ent_coef)
        tf.summary.scalar('Value loss', vf_loss * vf_coef)
        tf.summary.scalar('Diversity', div)
        tf.summary.scalar('Reward', tf.reduce_mean(R))
        merged = tf.summary.merge_all()
        print('log_dir:', log_dir)
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model") # find backup weights
        print("------Print out all trainable variables------")
        for p in params:
            print(p)

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            # for t in grads:
            #     n = t.name
            #     norm = tf.sqrt(tf.reduce_sum(tf.norm(t, ord=2)**2))
            #     tf.summary.scalar(n, norm)

            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)
        # _backup_copy = trainer.backup_copy()

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values, update):
            # train for a single step
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values      

            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, policy_diversity, summary, _ = sess.run(
                [pg_loss, vf_loss, entropy, div, merged, _train],
                td_map  # feed_dict
            )
            writer.add_summary(summary, update)

            return policy_loss, value_loss, policy_entropy, policy_diversity

        self.bm_counter = 0     # Count which back up model to save
        def copy_to_backup_model():
            # get current policy weights
            cur_params = tf.trainable_variables('a2c_model')

            # get backup policy weights
            backup_params = tf.trainable_variables('a2c_backup_model_{}'.format(self.bm_counter))

            copy_op = []
            for c, b in zip(cur_params, backup_params):
                c_name = "/".join(c.name.split('/')[1:])
                b_name = "/".join(b.name.split('/')[1:])
                assert c_name == b_name, "Error raise when copy current policy to backup."
                copy_op.append(tf.assign(b, c))

            # Update bm_counter
            self.bm_counter = self.bm_counter + 1 if self.bm_counter < 4 else 0

            sess.run(copy_op)
            # print('assign weights to backup {}'.format(self.bm_counter))            

        self.copy_to_backup_model = copy_to_backup_model  # HIGHLIGHT
        self.train = train      # HIGHLIGHT
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)    # HIGHLIGHT
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(      # Call Model inside the learn function
    network,
    env,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    backup_interval=20,
    load_path=None,
    log_dir=None,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)
    # I have to wrapped env


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)
    # Remaining coef is policy gradient loss

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    backup_interval     int, specifies how frequently save policies for comparing the KL divergence. (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''

    set_global_seeds(seed)  # Existing seed makes experiment reproducable

    # Build dir to save tensorboard data if log_dir not yet exist
    build_log_dir(log_dir)

    # Get the nb of env
    nenvs = env.num_envs

    # Forward policy
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model, train_model and backup models)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, log_dir=log_dir,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    # add back up policy, sightly change
    # model define loss, whereas policy define forward propagation.

    # Load trained model
    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)     # Runner get training example batches
    # add backup_policys steps in runner 

    # Calculate the batch_size
    nbatch = nenvs*nsteps

    # Start total timer
    tstart = time.time()

    for update in range(1, total_timesteps//nbatch+1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values = runner.run()
        # value generate by model.step
        # reward comes from env

        # model.train() update parameters one time
        policy_loss, value_loss, policy_entropy, policy_diversity = model.train(obs, states, rewards, masks, actions, values, update)
        # add backup_step policy to arguments of model.trains

        nseconds = time.time()-tstart

        # Calculate the fps (frame per second)
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)   # from baselines import logger
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_diversity", float(policy_diversity))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()

        # copy weights into backup model
        if update % backup_interval == 0:
            model.copy_to_backup_model()    # save current policy into one of the backup policy. 
            

        # if 
    return model


def build_log_dir(log_dir=None):
    if log_dir:
        import os

        # assert !(os.path.ispath(log_dir)), 'log_dir already existed.'

        os.makedirs(log_dir, exist_ok=True)
        print('Tensorboard info save in path:', log_dir)

    