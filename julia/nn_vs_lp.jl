

function nn_vs_lp(layers::Vector{Any}, model::Model)
    
    # optimize the model
    optimize!(model)

    # father the optimal value of the input variable y
    y_model = Array(value.(model[:y]))

    # get the dimentions of each layer
    n_of_neurons = layers_to_n_of_neurons(layers)

    # gather the optimal values of the output neurons z
    z_model = Array{Any}(undef, n_of_neurons[end])
    for i=1:n_of_neurons[end]
        z_model[i] = value.(model[:z][length(n_of_neurons),i])
    end

    # calculate the output of the input convex network given the input y_model
    z_nn = Array{Any}(undef, length(n_of_neurons))
    for i = 1:length(n_of_neurons)
        if i == 1
            # if it is the input layer, calculate the output using the relu activation function z_i = max(y*A_i + b+i,0)
            z_nn[i] = max.(y_model*layers[i].A .+ layers[i].b,0)
        else
            # if it is not the input layer, calculate the output using the relu activation function for z_i = max(y*A_i + b_i + z_{i-1}*W_i, 0)
            z_nn[i] = max.(y_model*layers[i].A .+ layers[i].b .+ z_nn[i-1]*layers[i].W, 0)
        end
    end
    
    # calculate the mean squared error between the output of the input convex network and the output of the linear programming model
    msqe = 1/length(z_model)*(sum((z_nn[end]' .- z_model).^2))

    return msqe


end
