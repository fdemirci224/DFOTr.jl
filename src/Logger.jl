# Logger.jl - Logging utility for DFO-TR

module Logger

using Printf

export IterationLogger, log_iteration, save_log

mutable struct IterationLogger
    data::Vector{Dict{String, Any}}
    verbosity::Int
    
    function IterationLogger(verbosity::Int=0)
        new(Vector{Dict{String, Any}}(), verbosity)
    end
end

"""
    log_iteration(logger, iter_data)

Log iteration data if verbosity >= 2.
"""
function log_iteration(logger::IterationLogger, iter_data::Dict{String, Any})
    if logger.verbosity >= 2
        push!(logger.data, copy(iter_data))
    end
end

"""
    save_log(logger, filename)

Save logged data to CSV or JSON file.
"""
function save_log(logger::IterationLogger, filename::String)
    if isempty(logger.data)
        return
    end
    
    if endswith(filename, ".csv")
        save_csv(logger, filename)
    elseif endswith(filename, ".json")
        save_json(logger, filename)
    else
        @warn "Unsupported file format. Use .csv or .json"
    end
end

function save_csv(logger::IterationLogger, filename::String)
    if isempty(logger.data)
        return
    end
    
    # Get all unique keys
    all_keys = Set{String}()
    for entry in logger.data
        union!(all_keys, keys(entry))
    end
    sorted_keys = sort(collect(all_keys))
    
    open(filename, "w") do file
        # Write header
        println(file, join(sorted_keys, ","))
        
        # Write data rows
        for entry in logger.data
            # Convert values to strings and CSV-escape when needed
            vals_str = [string(get(entry, key, "")) for key in sorted_keys]
            escaped = map(vals_str) do s
                needs_quotes = occursin(",", s) || occursin('\n', s) || occursin('"', s)
                if needs_quotes
                    '"' * replace(s, '"' => "\"\"") * '"'
                else
                    s
                end
            end
            println(file, join(escaped, ","))
        end
    end
    
    println("Iteration log saved to: $filename")
end

function save_json(logger::IterationLogger, filename::String)
    # Simple JSON output without external dependencies
    open(filename, "w") do file
        println(file, "[")
        for (i, entry) in enumerate(logger.data)
            print(file, "  {")
            keys_vals = collect(pairs(entry))
            for (j, (key, val)) in enumerate(keys_vals)
                print(file, "\"$key\": ")
                if isa(val, String)
                    print(file, "\"$val\"")
                else
                    print(file, "$val")
                end
                if j < length(keys_vals)
                    print(file, ", ")
                end
            end
            print(file, "}")
            if i < length(logger.data)
                println(file, ",")
            else
                println(file)
            end
        end
        println(file, "]")
    end
    
    println("Iteration log saved to: $filename")
end

end # module
