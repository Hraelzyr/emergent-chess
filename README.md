# Emergent Chess

Making an actor teach another the way to play chess through examples and RL.

## Models

An RNN is used for speaker and listener and a simple linear critic.

## Methods

As of now, REINFORCE with critic baseline (simple actor-critic) method is present.  
Huber loss is used for critic and policy gradient loss conditioned on advantage for the actor.

## Tests

The shape of outputs and behaviour of reward functions is checked.
