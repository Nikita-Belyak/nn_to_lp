def sample_data_gen(DATASET, BATCH_SIZE):
	
	if DATASET == 'styblinski_tang':

		# Specify the range
		lower_bound = -5
		upper_bound = 5

		# Specify the size of the vector
		vector_size = opt.INPUT_DIM  # Adjust this to the desired size
		
		while True:
			dataset_x = []
			dataset_y = []
	
			for i in range(BATCH_SIZE):
				
				point = np.random.uniform(lower_bound, upper_bound, size=vector_size)
					
				# Define the expression x^4 - 16x^2 + 5x
				expression_result = point**4 - 16 * point**2 + 5 * point

				# Compute the sum of the expression
				result_sum = np.sum(expression_result)

				dataset_x.append(point)
				dataset_y.append(result_sum)

			dataset_x = np.array(dataset_x, dtype='float32')
			dataset_y = np.array(dataset_y, dtype='float32')
			yield dataset_x, dataset_y

	elif DATASET == 'sphere_function':

		# Specify the range
		lower_bound = -1000
		upper_bound = 1000

		# Specify the size of the vector
		vector_size = opt.INPUT_DIM  # Adjust this to the desired size
		
		while True:
			dataset_x = []
			dataset_y = []
	
			for i in range(BATCH_SIZE):
				
				point = np.random.uniform(lower_bound, upper_bound, size=vector_size)
					
				# Define the expression x^4 - 16x^2 + 5x
				expression_result = point**2

				# Compute the sum of the expression
				result_sum = np.sum(expression_result)

				dataset_x.append(point)
				dataset_y.append(result_sum)

			dataset_x = np.array(dataset_x, dtype='float32')
			dataset_y = np.array(dataset_y, dtype='float32')
			yield dataset_x, dataset_y
            
            

