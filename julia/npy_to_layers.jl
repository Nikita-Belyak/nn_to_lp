using NPZ

"""
    npy_to_layers(filename::AbstractString)

Converts a NumPy file containing layer data into a Julia array of layers.

# Arguments
- `filename::AbstractString`: The path to the NumPy files.

# Returns
- An array of layers, where each layer is represented as a Julia array.

"""

function npy_to_layers(data_folder_path::AbstractString)

    # Get a list of file names in the folder
    file_names = readdir(data_folder_path)

    # Display the file names
    println("Files in the folder:")
    for file in file_names
        println(file)
    end


    # fetch the maximum number from the file names
    layers_num = find_max_number(file_names) 

    # initatialize the layers array
    layers = []

    #fill the layers array with the data from the .npy files
    for l in 0:layers_num
        # load the data from the .npy files
        data_A = npzread(data_folder_path*"/layer$l"*"_matrix_A.npy")
        data_b = npzread(data_folder_path*"/layer$l"*"_matrix_b.npy")
    
        if l>0 # if the layers is not the input layer, load the W matrix
            data_W = npzread(data_folder_path*"/layer$l"*"_matrix_W.npy")
            # create a new layer with the data
            new_layer = layer(l, data_A, data_W, data_b)
        else
            # if the layer is the input layer, initialize the W matrix as an empty array
            new_layer = layer(l, data_A, zeros(Float32,0,0), data_b)
        end
        # push the new layer to the layers array
        push!(layers, new_layer)
    end
    return layers
end



"""
    find_max_number(file_names)

Find the maximum numerical value in a list of file names.

# Arguments
- `file_names::Vector{String}`: A vector of file names.

# Returns
- `max_number::Int`: The maximum numerical value found in the file names.
"""

function find_max_number(file_names::Vector{String})

    # Initialize max_number to a very small value
    max_number = typemin(Int)

    # Iterate through each file name
    for file_name in file_names

        # Regular expression to match the first numeric value
        regex = r"\d+(\.\d+)?"

        # Find the first numerical value using a regular expression
        numbers = match(regex, file_name)

        if numbers !== nothing
            # Extract and parse the numerical value
            parsed_number = parse(Int, numbers.match)

            # Update max_number if a larger numerical value is found
            max_number = max(max_number, parsed_number)
        end
        
    end

    return max_number
end


"""
    struct layer

A mutable struct representing a layer.

# Fields
- `num::Int64`: The layer number.
- `A::Matrix{Float32}`: The matrix A.
- `W::Matrix{Float32}`: The matrix W.
- `b::Matrix{Float32}`: The matrix b.
"""

mutable struct layer
    num::Int64
    A::Matrix{Float32}
    W::Matrix{Float32}
    b::Matrix{Float32}
end




