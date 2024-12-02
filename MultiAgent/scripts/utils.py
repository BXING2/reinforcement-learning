#
import torch

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs import TransformedEnv, RewardSum, Compose, StepCounter

from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

from torchrl.collectors import SyncDataCollector

from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.objectives import ClipPPOLoss, ValueEstimators

class MultiAgent:
    '''
    Uni-Agent class 
    '''    
    def __init__(
        self,
        args,   # simulation arguments  
    ):

        # load simulation arguments     
        self.args = args 

        # build actor module
        self.actor_module = self.build_actor()

        # build critic module 
        self.critic_module = self.build_critic()

    def build_actor(
        self,
    ):

        # build actor neural network

        # model: observation --> mean/standard deviation of action distrituion 
        actor_model = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=self.args["obs_dims"],
                n_agent_outputs=self.args["action_dims"]*2, # mean and standard deviation 
                n_agents=self.args["n_agents"],
                centralised=self.args["centralised_actors"],
                share_params=self.args["shared_actors"],
                depth=self.args["actor_depth"],
                num_cells=self.args["actor_width"],
                activation_class=self.args["actor_acti_func"],
                device=self.args["device"],
            ),
            NormalParamExtractor(), # extract mean and standard deviation from output layer 
        )

        # wrap actor model into tensordict module
        actor_module = TensorDictModule(
            actor_model, # torch nn module
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
   
        # build probabilistic actor module 
        actor_module = ProbabilisticActor(
            module=actor_module,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[("agents", "action")],
            distribution_class=TanhNormal, # action value between -1 and 1, tanh loc default is false
            return_log_prob=True, # add log probability of the sampled action 
            log_prob_key=("agents", "log_prob"), 
        ) 

        return actor_module

    def build_critic(
        self,
    ):
        # method for building critic network        

        # build critic neural network
        critic_model = MultiAgentMLP(
            n_agent_inputs=self.args["obs_dims"],
            n_agent_outputs=1, # state value,
            n_agents=self.args["n_agents"],
            centralised=self.args["centralised_critics"],
            share_params=self.args["shared_critics"],
            depth=self.args["critic_depth"],
            num_cells=self.args["critic_width"],
            activation_class=self.args["critic_acti_func"],
            device=self.args["device"],
        )

        # wrap critic in tensordict module
        critic_module = TensorDictModule(
            module=critic_model,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )
 
        return critic_module

def load_env(args):
    # function for building environments

    # load base environment from VMAS
    env = VmasEnv(
        scenario=args["env_name"],
        continuous_actions=True,
        num_envs=args["n_envs"],
        max_steps=args["max_steps"],
        n_agents=args["n_agents"],
        device=args["device"],
    )

    # apply transforms to environment based on user's need
    env = TransformedEnv(
        env,
        Compose(
            StepCounter(),
            RewardSum(
                in_keys=[("agents", "reward")],
                out_keys=[("agents", "episode_reward")],
            )
        ),
    )

    return env


def load_collector(
    args, # args related to environment
    envs, # environment
    actor, # actor for selecting actions given states,
):
 
    # load collector for collecting MDP traj 
    collector = SyncDataCollector(
        envs,
        actor,
        frames_per_batch=args["frames_per_batch"],
        total_frames=args["total_frames"],
        device=args["device"],
    )

    return collector 

def load_buffer(
    args,
):

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            args["frames_per_batch"],
            device=args["device"] ,           
        ),
        sampler=SamplerWithoutReplacement(),
        batch_size=args["minibatch_size"],
    )

    return replay_buffer 

def load_loss(
    args, # training arguments 
    actor, # actor module 
    critic, # critic module
):
    
    # build loss module 
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=args["clip_epsilon"],
        entropy_coef=args["entropy_coef"],
        normalize_advantage=args["normalize_advantage"],
    )

    # set up keys
    loss_module.set_keys(
        reward=("agents", "reward"),
        action=("agents", "action"),
        sample_log_prob=("agents", "log_prob"), # from actor module
        value=("agents", "state_value"), # from critic module
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    # note: the done and terminated are global in mappo settings and will be added to the agents.

    # Build value estimator
    loss_module.make_value_estimator(
        ValueEstimators.GAE, 
        gamma=args["gamma"],
        lmbda=args["lmbda"],
    )

    # Build optimizer
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=args["learning_rate"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        args["total_frames"] // args["frames_per_batch"], # number of iterations
        0.0, # minimum learning rate 
    )

    return loss_module, optim, scheduler

def preprocess(batch):

    target_dims = batch[("next", "agents", "reward")].shape

    batch.set(
        ("next", "agents", "done"),
        batch[("next", "done")] .unsqueeze(-1).expand(target_dims)
    )

    batch.set(
        ("next", "agents", "terminated"),
        batch[("next", "terminated")] .unsqueeze(-1).expand(target_dims)
    )
 

def compute_advantage(
    batch,
    loss,
):
    
    with torch.no_grad():
        loss.value_estimator(
            batch,
            params=loss.critic_network_params,
            target_params=loss.target_critic_network_params,   
        )

def postprocess(
    i_iteration,
    batch,
    actor,
    metrics,
    save_freq,
    f,
):
    
    agent_done = batch[("next", "agents", "done")]
    episode_reward = batch[("next", "agents", "episode_reward")]
    
    done = batch[("next", "done")]
    step_count = batch[("next", "step_count")]

    # ave episode reward and step count 
    episode_reward_mean = episode_reward[agent_done].mean().item()
    step_count_mean = step_count[done].mean(dtype=torch.float32).item() 

    print(i_iteration, episode_reward_mean, step_count_mean, file=f)
    
    # save actor network
    if (i_iteration+1) % save_freq == 0: 
        torch.save(actor.state_dict(), "actor_{}.pt".format(i_iteration+1))
    
    return metrics.append([episode_reward_mean, step_count_mean])
