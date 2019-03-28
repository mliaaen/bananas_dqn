# Getting started with bananas_dqn

This is the report from the Udacity Deep Reinforcement course Project: Navigation

1. create a virtual environment running python3.
2. source the environment
3. install the requirements 
4. pip install -r requirements.txt



# The Environment for collecting bananas

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

  * 0 - move forward
  * 1 - move backward
  * 2 - turn left
  * 3 - turn right

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Installing the environment

For this project, you will not need to install Unity - this is because the environment is already built, and you can download it from one of the links below. You need only select the environment that matches your operating system:

 * Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
 * Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
 * Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip

Then, place the file in the root of the source tree of bananas_dqn/bananas, and unzip (or decompress) the file.

## Solving the environment

The requriement is condisered solved when you reach an average of 13 or better over the 100 last episodes.


# Running the code
After making the virtual environment and the copied and uncompressed the collecting bananas unity environment to the right place, you are
ready to train the agents.

The jupyter notebook Bananas.ipynb contains the training and visualtion code for solving this project.
There are two modules that are imported.

 1. dqn_agent.py contains the DQN and DDQN agent
 2. model.py contains the neural network

The notebook trains both a DQN and a DDQN version of the agent. A demo of a trained agent is shown at the end of the notebook.
An samle output from a run on my Macbook Pro is shown in Bananas_run.pdf