from csv import reader
import multiprocessing as mp

# Load a CSV file
def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm 
def evaluate_algorithm(train_set,test_set, algorithm, *args):
    predicted = algorithm(train_set, test_set, *args)
    actual = [row[-1] for row in test_set]
    accuracy = accuracy_metric(actual, predicted)
    
    return accuracy

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)


def main():
    # load and prepare data
    data_train_set = 'D:\data_train.csv'
    data_test_set = 'D:\data_test.csv'
    train_set = load_csv(data_train_set)
    test_set = load_csv(data_test_set)
   
    #data to float
    for i in range(len(train_set[0])):
        str_column_to_float(train_set, i)
    for i in range(len(test_set[0])):
        str_column_to_float(test_set, i)

    min_size = 1
   # max_depth = 5
    #Accuracy = evaluate_algorithm(train_set,test_set, decision_tree, max_depth, min_size)
    #print('Accuracy: %.3f%%' % Accuracy)
    """
    procs = list()
    depth = [1,3,5,7]
    
    for i in depth:
        proc = mp.Process(target = evaluate_algorithm, args = (train_set, test_set, decision_tree, i, min_size))
        procs.append(proc)
        proc.start()
        
    for proc in procs:
        proc.join()
        """
    p=mp.Pool(4) #processes pool
    args=[[train_set, test_set, decision_tree, 1, min_size],[train_set, test_set, decision_tree, 3, min_size],
             [train_set, test_set, decision_tree, 5, min_size],[train_set, test_set, decision_tree, 7, min_size]]
    for i in args:
        print( "accuracy ", p.apply(evaluate_algorithm,i), "with depth",i[-2])
    
if __name__ == '__main__':
    main()