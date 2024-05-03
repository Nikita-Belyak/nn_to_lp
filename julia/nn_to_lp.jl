

using JuMP
using Gurobi 

"""
    icnn_to_lp(nn)

Converts a input convex neural network `nn` to a linear programming (LP) representation.

# Arguments
- `nn::Vector{layer}`: The neural network to be converted.

# Returns
- `lp::JuMP.LinearProgram`: The linear programming representation of the neural network.

"""
function icnn_to_lp(layers::Vector{Any})

    n_of_neurons = layers_to_n_of_neurons(layers)

    # Use @model to begin creating a JuMP model
    model = Model(Gurobi.Optimizer)

    # Use the @variable macro to create the variables
    @variable(model, z[j=1:length(n_of_neurons), i = 1:n_of_neurons[j]])

    @variable(model, y[1,i = 1:size(layers[1].A,1)])

    # Add the constraints
    for i = 0:length(n_of_neurons)-1
    
        if i==0 # if it is the input layer
            # generate right-had side of the constraints: rhs = y*A_1 + b_1
            rhs = model[:y]*layers[i+1].A .+ layers[i+1].b
            # add the constraints correspoding to the relu activation function: that is z_1 >= max(0, rhs)
            @show n_of_neurons[i+1]
            @constraint(model, [n = 1:n_of_neurons[i+1]], z[i+1,n] >= rhs[1,n])
            @constraint(model, [n = 1:n_of_neurons[i+1]], z[i+1,n] >= 0)
        else 
            # generate the mutiplication z_i*W_i+1
            zw = [sum(z[i,k]* layers[i+1].W[k,j] for k = 1:size(layers[i+1].W,1) ) for j = 1:size(layers[i+1].W,2)]
            zw = zw' # make it a row vector 
            #generate the right-hand side of the constraints: rhs = y*A_i+1 + b_i+1 + z_i*W_i+1
            rhs = model[:y]*layers[i+1].A .+ layers[i+1].b .+ zw
            # add the constraints correspoding to the relu activation function: that is z_i+1 >= max(0, rhs)
            @constraint(model, [n = 1:n_of_neurons[i+1]], z[i+1,n] >= rhs[1,n])
            @constraint(model, [n = 1:n_of_neurons[i+1]], z[i+1,n] >= 0)
        end 
   
    end 
    # Add the objective function as maximise the sum of the last layer values
    @objective(model, Min, sum(z[length(n_of_neurons),1:n_of_neurons[end]]))

    return model

end


function nn_to_lp(layers::Vector{Any})

    n_of_neurons = layers_to_n_of_neurons(layers)

    # Use @model to begin creating a JuMP model
    model = Model(Gurobi.Optimizer)

    # Use the @variable macro to create the variables
    @variable(model, z[j=1:length(n_of_neurons), i = 1:n_of_neurons[j]])

    @variable(model, y[1,i = 1:size(layers[1].W,1)])
    @show layers[1].W' .* model[:y] .+ layers[1].b
    # Add the constraints
    for i = 0:length(n_of_neurons)-1
    
        if i==0 # if it is the input layer
            # generate right-had side of the constraints: rhs = y*A_1 + b_1
            # rhs = layers[i+1].W' .* model[:y] .+ layers[i+1].b
            # @show rhs[1,:][2]
            # add the constraints correspoding to the relu activation function: that is z_1 >= max(0, rhs)
            # @show n_of_neurons[i+1]
            # @show layers[i+1].W'[1,:] .* model[:y] .+ layers[i+1].b[1,:]
            @constraint(model, [n = 1:n_of_neurons[i+1]], z[i+1,n] .>= model[:y] .* layers[i+1].W[:,n] .+ layers[i+1].b[n,:])
            @constraint(model, [n = 1:n_of_neurons[i+1]], z[i+1,n] >= 0)
        else 
            # generate the mutiplication z_i*W_i+1
            zw = [sum(z[i,k]* layers[i+1].W[k,j] for k = 1:size(layers[i+1].W,1) ) for j = 1:size(layers[i+1].W,2)]
            zw = zw' # make it a row vector 
            #generate the right-hand side of the constraints: rhs = y*A_i+1 + b_i+1 + z_i*W_i+1
            rhs = layers[i+1].b .+ zw
            # add the constraints correspoding to the relu activation function: that is z_i+1 >= max(0, rhs)
            @constraint(model, [n = 1:n_of_neurons[i+1]], z[i+1,n] >= rhs[n,1])
            @constraint(model, [n = 1:n_of_neurons[i+1]], z[i+1,n] >= 0)
        end 
   
    end 
    # Add the objective function as maximise the sum of the last layer values
    @objective(model, Min, sum(z[length(n_of_neurons),1:n_of_neurons[end]]))

    return model

end

"""
layers_to_n_of_neurons(layers::Vector{Any})

Converts a vector of layer sizes of each layer in the network.

# Arguments
- `layers::Vector{Any}` - A vector of layers.

# Returns
- A vector of integers representing the sizes of each layer in the network.


"""

function layers_to_n_of_neurons(layers::Vector{Any})

    # Get the number of neurons in each layer
    n_of_neurons = [size(layers[i].W,1) for i = 2:length(layers)]

    # add the last layer (with one neuron z + y*A)
    push!(n_of_neurons, max(1, size(layers[end].A,2)))

    return n_of_neurons
end
