#--------------------------------------------
# NAME: SEUN OMOJOLA
# STUDENT NUMBER: 7880480 
# COURSE: COMP 4820
#---------------------------------------------
import pandas as pd
import sys
import numpy as np


# Step 1: Parse command line argument
emission_file = sys.argv[1]
transition_file = sys.argv[2]
output_sequence = sys.argv[3]


# Step 2: Parse the emission probability matrix file
def parse_emission_matrix(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    header = lines[0].strip().split('\t')[0:] 
    data = [line.strip().split('\t') for line in lines[1:]]  
    row_headers = [row[0] for row in data]
    probabilities = [[float(prob) for prob in row[1:]] for row in data]

    return probabilities, header, row_headers



# Step 3: Parse the transition probability matrix file
def parse_transition_matrix(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        
    header = lines[0].strip().split('\t')[0:]  
    data = [line.strip().split('\t') for line in lines[1:]]  
    row_headers = [row[0] for row in data]
    probabilities = [[float(prob) for prob in row[1:]] for row in data]

    return probabilities, header, row_headers

    

# Step 4: Implement the Viterbi algorithm
def viterbi_algorithm(emission_probs, emission_row_headers, transition_probs, output_sequence, transition_row_headers,emission_headers):
    # Create a pandas DataFrame for the dynamic programming table
    num_states = len(emission_row_headers)
    sequence_length = len(output_sequence)
    list_seq = list(output_sequence)
    pd.set_option("display.precision", 12)

    
    column_range = range(1, len(list_seq) + 1)
    column_header = [str(i) + ':' + header for i, header in zip(column_range, list_seq)]
    column_header.insert(0,"0")
    list_seq.insert(0,"0")
    viterbi_table = pd.DataFrame(index=transition_row_headers, columns=column_header)
    
    for i in range(len(transition_row_headers)):
        for j in range(len(list_seq)):
            viterbi_table.iloc[i,j] = 0
    viterbi_table.iloc[0,0] = 1
    
    
    # Fill the Viterbi table    
    for i in range(1, sequence_length+1):
        indx = emission_headers.index(output_sequence[i-1])
        for s in range(1, num_states+1):
            for k in range(num_states+2):
                current_value = viterbi_table.iloc[k].iloc[i-1] * transition_probs[k][s] * emission_probs[s-1][indx]
                if current_value > viterbi_table.iloc[s].iloc[i]:
                    viterbi_table.iloc[s].iloc[i] = current_value
                if viterbi_table.iloc[s].iloc[i] == 0:
                    viterbi_table.iloc[s].iloc[i] = 0.0
 
 
     # Add the end state value
    max_E = 0
    for l in range(num_states + 1):
        current_value = viterbi_table.iloc[l].iloc[sequence_length] * transition_probs[l][num_states+1]  
        if current_value > max_E:                 
            viterbi_table.iloc[num_states+1].iloc[sequence_length] = current_value  
            max_E = current_value                 
            
    
    # Find the best path
    best_path = []
    max_last_state = -1 
    if(viterbi_table.iloc[num_states+1].iloc[sequence_length] > 0):
        best_path.append(viterbi_table.index[num_states+1])
    for i in range(sequence_length,-1,-1):
        max = 0        
        for j in range(num_states+2):
            if viterbi_table.iloc[j].iloc[i] > max:
                max = viterbi_table.iloc[j].iloc[i]
                max_last_state = viterbi_table.index[j]
        best_path.append(max_last_state)
        max_last_state = -1
        
    return best_path, viterbi_table


# Step 5: Output and display
emission_probs, emission_headers, emission_row_headers = parse_emission_matrix(emission_file)
transition_probs, transition_headers, transition_row_headers = parse_transition_matrix(transition_file)

best_path, viterbi_table = viterbi_algorithm(emission_probs, emission_row_headers, transition_probs, output_sequence, transition_row_headers,emission_headers)

    # Output the best path
print("Best_path =", end = " ")
for i in range(len(best_path)-1,0,-1):
    print(best_path[i], end = " -> ")
print(best_path[0])
print("\n ########## TABLE ########## \n")
print(viterbi_table)