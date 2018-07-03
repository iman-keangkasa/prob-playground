import comp_prob_inference

# PROBABILITY SPACES AND EVENTS
#def prob_of_event(event, prob_space):
#	total = 0
#	for outcome in event:
#		total += prob_space[outcome]
#	return total

#RANDOM VARIABLES EXERCISE

prob_space = {'sunny': 1./2, 'rainy':1./6, 'snowy': 1./3}
random_outcome = comp_prob_inference.sample_from_finite_probability_space(prob_space)
W = random_outcome
if random_outcome == 'sunny':
	I = 1
else:
	I = 0	

