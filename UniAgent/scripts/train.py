#
import torch 
import params, utils

# train uni-agent system using ppo algorithm 
def train():

    # --- initialize params --- #
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    
    args = params.args()

    # simulation environment
    args.add_args("env_name", "InvertedDoublePendulum-v4")

    # batch settings
    args.add_args("n_iterations", 100)
    args.add_args("frames_per_batch", 1000)
    args.add_args("minibatch_size", 100)
    args.add_args("total_frames", args.args["n_iterations"] * args.args["frames_per_batch"])
    args.add_args("n_minibatches", args.args["frames_per_batch"] // args.args["minibatch_size"])

    # actor model settings
    args.add_args("actor_depth", 3) 
    args.add_args("actor_width", 256)
    args.add_args("actor_acti_func", torch.nn.Tanh)
     
    # critic model settings
    args.add_args("critic_depth", 3)
    args.add_args("critic_width", 256)
    args.add_args("critic_acti_func", torch.nn.Tanh)

    # loss functions settings
    args.add_args("gamma", 0.99)
    args.add_args("lmbda", 0.9)
    args.add_args("normalize_advantage", True)
    args.add_args("clip_epsilon", 0.2)
    args.add_args("entropy_coef", 1e-4)

    # training settings
    args.add_args("device", device)
    args.add_args("n_epochs", 10)
    args.add_args("learning_rate", 3e-4)
    args.add_args("max_grad_norm", 1.0)
    args.add_args("save_freq", 10)
    

    # --- load simulation environment --- #
    envs = utils.load_env(args.args)
   
    args.add_args("obs_dims", envs.observation_spec["observation"].shape[-1]) # add observation dims
    args.add_args("action_dims", envs.action_spec.shape[-1]) # add action dims

    # check args 
    args.print_args()

    # --- load agent --- #
    agent = utils.UniAgent(
        args.args,
    )
    
    # --- load data collector --- #
    collector = utils.load_collector(
        args=args.args,
        envs=envs,
        actor=agent.actor_module,
    )

    # --- load data buffer --- #
    replay_buffer = utils.load_buffer(
        args=args.args,
    )

    # --- load loss and optimizer --- #
    loss_module, optim, scheduler = utils.load_loss(
        args=args.args,
        actor=agent.actor_module,
        critic=agent.critic_module,
    )
    
    # ---- train agent --- # 
    metrics = []
    f = open("log.txt", "w")
    for i, batch in enumerate(collector):
        
        # preprocess data
        utils.preprocess(batch)

        # compute advantage
        utils.compute_advantage(
            batch,
            loss_module,
        )
        
        # add data to replay buffer
        replay_buffer.extend(batch.reshape(-1))
        
        # train multiple epochs for each iteration 
        for _ in range(args.args["n_epochs"]):
            # minibatch training 
            for _ in range(args.args["n_minibatches"]):
                # sample minibatch data
                minibatch = replay_buffer.sample()
                
                # compute loss 
                loss = loss_module(minibatch)
                loss = loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]

                # backpropagation
                optim.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(),
                    args.args["max_grad_norm"]
                )
                
                optim.step()
                
                # end of minibatch
            # end of epoch
        # end of batch 
    
        # postprocess
        utils.postprocess(
            i, # ith iteration
            batch, # batch data
            agent.actor_module, # actor model
            metrics, # save metrics 
            args.args["save_freq"], # save every freq
            f,
        )

        scheduler.step()
    # end of training
    
    f.close()  # close log file

def main():
    train()

if __name__ == "__main__":
    main()
