import comp_prob_inference
import numpy as np
import matplotlib.pyplot as plt
########################################################################
#               ____            _           _     _ _ _ _         
#              |  _ \ _ __ ___ | |__   __ _| |__ (_) (_) |_ _   _ 
#              | |_) | '__/ _ \| '_ \ / _` | '_ \| | | | __| | | |
#              |  __/| | | (_) | |_) | (_| | |_) | | | | |_| |_| |
#              |_|   |_|  \___/|_.__/ \__,_|_.__/|_|_|_|\__|\__, |
#                                                           |___/ 
#                                                               _ 
#               ___ _ __   __ _  ___ ___  ___    __ _ _ __   __| |
#              / __| '_ \ / _` |/ __/ _ \/ __|  / _` | '_ \ / _` |
#              \__ \ |_) | (_| | (_|  __/\__ \ | (_| | | | | (_| |
#              |___/ .__/ \__,_|\___\___||___/  \__,_|_| |_|\__,_|
#                  |_|                                            
#                                               _       
#                           _____   _____ _ __ | |_ ___ 
#                          / _ \ \ / / _ \ '_ \| __/ __|
#                         |  __/\ V /  __/ | | | |_\__ \
#                          \___| \_/ \___|_| |_|\__|___/
                                                       
#######################################################################
#probability spaces and events

#def prob_of_event(event, prob_space):
#	total = 0
#	for outcome in event:
#		total += prob_space[outcome]
#	return total

######################################################################
#                     ____                 _                 
#                    |  _ \ __ _ _ __   __| | ___  _ __ ___  
#                    | |_) / _` | '_ \ / _` |/ _ \| '_ ` _ \ 
#                    |  _ < (_| | | | | (_| | (_) | | | | | |
#                    |_| \_\__,_|_| |_|\__,_|\___/|_| |_| |_|
#                                                            
#                  __     __         _       _     _           
#                  \ \   / /_ _ _ __(_) __ _| |__ | | ___  ___ 
#                   \ \ / / _` | '__| |/ _` | '_ \| |/ _ \/ __|
#                    \ V / (_| | |  | | (_| | |_) | |  __/\__ \
#                     \_/ \__,_|_|  |_|\__,_|_.__/|_|\___||___/
#                                                              
######################################################################
#Random variables

#prob_space = {'sunny': 1./2, 'rainy':1./6, 'snowy': 1./3} #DEFINING A PROBABILITY SPACE OR PROBABILITY MODEL
#random_outcome = comp_prob_inference.sample_from_finite_probability_space(prob_space) #GETTING THE RANDOM OUTCOME SMALL OMEGA
#W = random_outcome #MAPPING THE RANDOM VARIABLES AS ALL POSSIBLE VALUE OF THE RANDOM OUTCOME SMALL OMEGA 
#if random_outcome == 'sunny':#MAPPING THE RANDOM VARIABLE I FROM THE ORIGINAL PROBABILITY SAMPLE SPACE
#	I = 1					 #MAPPING THE RANDOM VARIABLE I FROM THE ORIGINAL PROBABILITY SAMPLE SPACE	
#else:						 #MAPPING THE RANDOM VARIABLE I FROM THE ORIGINAL PROBABILITY SAMPLE SPACE  
#	I = 0					 #MAPPING THE RANDOM VARIABLE I FROM THE ORIGINAL PROBABILITY SAMPLE SPACE to 1 and 0
#print "The value I is " + str(I)
#print "The value of W is " + W

#####################################################################################################
#                            _       _       _   _       
#                           | | ___ (_)_ __ | |_| |_   _ 
#                        _  | |/ _ \| | '_ \| __| | | | |
#                       | |_| | (_) | | | | | |_| | |_| |
#                        \___/ \___/|_|_| |_|\__|_|\__, |
#                                                  |___/ 
#         _ _     _        _ _           _           _   ______     ___     
#      __| (_)___| |_ _ __(_) |__  _   _| |_ ___  __| | |  _ \ \   / ( )___ 
#     / _` | / __| __| '__| | '_ \| | | | __/ _ \/ _` | | |_) \ \ / /|// __|
#    | (_| | \__ \ |_| |  | | |_) | |_| | ||  __/ (_| | |  _ < \ V /   \__ \
#     \__,_|_|___/\__|_|  |_|_.__/ \__,_|\__\___|\__,_| |_| \_\ \_/    |___/
#
#####################################################################################################
#jointly distributed random variables

#APPROACH 1: USING DICTIONARIES WITHIN DICTIONARY APPROACH
#prob_W_T_dict = {}
#for w in {'sunny', 'rainy', 'snowy'}:
#	prob_W_T_dict[w]={}

#prob_W_T_dict['sunny']['hot']=3./10
#prob_W_T_dict['sunny']['cold']=1./5
#prob_W_T_dict['rainy']['hot']=1./30
#prob_W_T_dict['rainy']['cold']=2./15
#prob_W_T_dict['snowy']['hot']=0
#prob_W_T_dict['snowy']['cold']=1./3
#print "The first approach uses dictionaries within dictionary"
#comp_prob_inference.print_joint_prob_table_dict(prob_W_T_dict)

#APPROACH 2: USING 2D ARRAYS
#prob_W_T_rows = ['sunny', 'rainy', 'snowy']
#prob_W_T_cols = ['hot','cold']
#prob_W_T_array=np.array([[3./10, 1./5], [1./30, 2./15], [0, 1./3]])
#print "The second approach uses 2D arrays"
#comp_prob_inference.print_joint_prob_table_array(prob_W_T_array, prob_W_T_rows, prob_W_T_cols)
#print "To retrieve an entry in the table we have to use index() function which is not efficient"
#print "The probability of 'cold' 'rainy' day is: "
#print prob_W_T_array[prob_W_T_rows.index('rainy')][prob_W_T_cols.index('cold')]
#using index to find a item in a list is less efficient because it searched through the whole list

#APPROACH 3: USING ARRAYS AND DICTIONARY (INDICES METHOD)
#prob_W_T_row_mapping = {}
#prob_W_T_row_mapping = {label: index for index, label in enumerate(prob_W_T_rows)}
#prob_W_T_col_mapping = {}
#prob_W_T_col_mapping = {label: index for index, label in enumerate(prob_W_T_cols)}
#print "By using labels remap to its indeces we will avoid using index()"
#print "The probability of a 'cold' 'rainy' day today is: "
#print prob_W_T_array[prob_W_T_row_mapping['rainy']][prob_W_T_col_mapping['cold']] 

#################################################################################################################
#                 ____  _                                     _ 
#                / ___|(_)_ __ ___  _ __  ___  ___  _ __  ___( )
#                \___ \| | '_ ` _ \| '_ \/ __|/ _ \| '_ \/ __|/ 
#                 ___) | | | | | | | |_) \__ \ (_) | | | \__ \  
#                |____/|_|_| |_| |_| .__/|___/\___/|_| |_|___/  
#                                  |_|                          
#                      ____                     _           
#                     |  _ \ __ _ _ __ __ _  __| | _____  __
#                     | |_) / _` | '__/ _` |/ _` |/ _ \ \/ /
#                     |  __/ (_| | | | (_| | (_| | (_) >  < 
#                     |_|   \__,_|_|  \__,_|\__,_|\___/_/\_\     An Exercise on joint probability dist.
##################################################################################################################
#Simpsson's paradox


#from simpsons_paradox_data import *
#print joint_prob_table[gender_mapping['female'],department_mapping['C'],admission_mapping['admitted']]

#joint_prob_gender_admission = joint_prob_table.sum(axis=1)
#female_only=joint_prob_gender_admission[gender_mapping['female']] #female only admission vector
#prob distribution given female 
#prob_admission_given_female = female_only/np.sum(female_only)
#print prob_admission_given_female

# Find the probability of admission given gender and department

#for dept in department_labels:
#	for gender in gender_labels:
#		prob_admission = joint_prob_table[gender_mapping[gender],department_mapping[dept]]
#
#		admission_sum=np.sum(prob_admission)
#		prob_admission_given_gd = prob_admission/admission_sum
#		print(dept, gender, dict(zip(admission_labels, prob_admission_given_gd))['admitted'])

#import numpy as np
#c = 1./72
#prob_space={}
#joint_prob_table=np.zeros((4,3))
#for x in range(1,5):
#	for y in range(1,4):
#		if x == 3:
##			prob_space[('x','y')]=0
#			joint_prob_table[x-1,y-1] = 0
#		if y == 2:
##			prob_space[('x','y')]=0
#			joint_prob_table[x-1,y-1] = 0
#		else:
#			joint_prob_table[x-1,y-1] = c*(x**2+y**2)
#print joint_prob_table
#x_table = joint_prob_table.sum(axis=1)
#print "X Distribution is: ", x_table
#y_table = joint_prob_table.sum(axis=0)
#print "Y Distribution is: ", y_table

#|_ _|_ __   __| | ___ _ __   ___ _ __   __| | ___ _ __   ___ ___         
# | || '_ \ / _` |/ _ \ '_ \ / _ \ '_ \ / _` |/ _ \ '_ \ / __/ _ \  _____ 
# | || | | | (_| |  __/ |_) |  __/ | | | (_| |  __/ | | | (_|  __/ |_____|
#|___|_| |_|\__,_|\___| .__/ \___|_| |_|\__,_|\___|_| |_|\___\___|        
 #                    |_|                                                 
#  ____                 _     _           _     
# / ___| __ _ _ __ ___ | |__ | | ___ _ __( )___ 
#| |  _ / _` | '_ ` _ \| '_ \| |/ _ \ '__|// __|
#| |_| | (_|=  | | | | | | |_) | |  __/ |    \__ \
# \____|\__,_|_| |_| |_|_.__/|_|\___|_|    |___/
#                                               
# _____     _ _                  
#|  ___|_ _| | | __ _  ___ _   _ 
#| |_ / _` | | |/ _` |/ __| | | |
#|  _| (_| | | | (_| | (__| |_| |
#|_|  \__,_|_|_|\__,_|\___|\__, |
#                          |___/ 
#Independece-gambler's fallacy

#n = np.array(range(1,100))
#plt.plot(n,1-(26./27)**(100-n))
#plt.xlabel('Number of rolls')
#plt.ylabel('Probability of seeing 27 at least once')
#plt.show()

#3D prob space for two succesive experiments on coins

#prob table = {
#	('head_up','fair','head2_up','fair'): 0.025,
#	('head_up','fair','head2_up','heads'): 0.1,
#	('head_up','heads','head2_up','fair'): 

#H_up_fair = 
#H_up_head = 
#H2_up_fair =  
#H2_up_head 


# ____            _     _                               _
#|  _ \  ___  ___(_)___(_) ___  _ __     __ _ _ __   __| |
#| | | |/ _ \/ __| / __| |/ _ \| '_ \   / _` | '_ \ / _` |
#| |_| |  __/ (__| \__ \ | (_) | | | | | (_| | | | | (_| |
#|____/ \___|\___|_|___/_|\___/|_| |_|  \__,_|_| |_|\__,_|

# _____                           _        _   _
#| ____|_  ___ __   ___  ___  ___| |_ __ _| |_(_) ___  _ __
#|  _| \ \/ / '_ \ / _ \/ _ \/ __| __/ _` | __| |/ _ \| '_ \
#| |___ >  <| |_) |  __/  __/ (__| || (_| | |_| | (_) | | | |
#|_____/_/\_\ .__/ \___|\___|\___|\__\__,_|\__|_|\___/|_| |_|
#           |_|Decision and expectation

#import random 
#sample = []
#for roll in range(9999):
#	sample.append(random.randint(1, 6))
	
#print np.mean(sample)

# ____  _
#/ ___|| |__   __ _ _ __  _ __   ___  _ __
#\___ \| '_ \ / _` | '_ \| '_ \ / _ \| '_ \
# ___) | | | | (_| | | | | | | | (_) | | | |
#|____/|_| |_|\__,_|_| |_|_| |_|\___/|_| |_|

# _____       _
#| ____|_ __ | |_ _ __ ___  _ __  _   _
#|  _| | '_ \| __| '__/ _ \| '_ \| | | |
#| |___| | | | |_| | | (_) | |_) | |_| |
#|_____|_| |_|\__|_|  \___/| .__/ \__, |
#                          |_|    |___/
#shannon entropy

entropy = lambda p: np.sum(p*np.log2(1/p))

#def entropy(p):
#	return np.sum(p*np.log2(1/p))

#entropy_L1 = entropy(np.array([999999./1000000, 1./1000000]))
#entropy_L2 = entropy(np.array([999999./1000000, 1./1000000]))
#entropy_L3 = entropy(np.array([9./10, 1./10]))
#print str(entropy_L1) + "\n" + str(entropy_L2) + "\n" + str(entropy_L3)

#p_list = np.linspace(0,1,50) #range 0 until 1 separated by 50 parts
#plt.figure()
#plt.plot(p_list,[entropy(np.array([p, 1-p])) for p in p_list])
#plt.xlabel('p')
#plt.ylabel('H(Ber(p)))')
#plt.show()

# __  __       _               _
#|  \/  |_   _| |_ _   _  __ _| |
#| |\/| | | | | __| | | |/ _` | |
#| |  | | |_| | |_| |_| | (_| | |
#|_|  |_|\__,_|\__|\__,_|\__,_|_|
#
# ___        __                            _   _
#|_ _|_ __  / _| ___  _ __ _ __ ___   __ _| |_(_) ___  _ __
# | || '_ \| |_ / _ \| '__| '_ ` _ \ / _` | __| |/ _ \| '_ \
# | || | | |  _| (_) | |  | | | | | | (_| | |_| | (_) | | | |
#|___|_| |_|_|  \___/|_|  |_| |_| |_|\__,_|\__|_|\___/|_| |_|
#mutual information

prob_XY = np.array([[1.10, 0.09, 0.11],[0.08,0.07,0.07],[0.18,0.13,0.17]])
prob_X = prob_XY.sum(axis = 1)
prob_Y = prob_XY.sum(axis = 0)
prob_indie_XY = np.outer(prob_X, prob_Y)

# my solution
#def compute_info(xy,indie_xy):
#	return xy*np.log2(xy/indie_xy)

#mutual_info = 0.0
#for x in range(len(prob_X)):
#	for y in range(len(prob_Y)):
#		mutual_info += compute_info(prob_XY[x,y],prob_indie_XY[x,y])

#print "The bit shared between X and Y are: " + str(mutual_info)

# edx solution

compute_info = lambda p, q: np.sum(p * np.log2(p / q))

print compute_info(prob_XY, prob_indie_XY)

