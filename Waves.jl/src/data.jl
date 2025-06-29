export Episode, generate_episode!, prepare_data

struct Episode{S, Y}
    s::Vector{S}
    a::Vector{<: AbstractDesign}
    t::Vector{Vector{Float32}}
    y::Vector{Y}
end

Base.length(ep::Episode) = length(ep.s)

function generate_episode!(policy::AbstractPolicy, env::WaveEnv; reset::Bool = true, position_mask::Union{AbstractArray, Nothing} = nothing)
    s = WaveEnvState[]
    a = AbstractDesign[]
    t = Vector{Float32}[]
    y = Matrix{Float32}[]

    if reset
        reset!(env)
    end
    
    while !is_terminated(env)
        push!(s, state(env))
        action = policy(env)
        push!(a, cpu(action))
        push!(t, build_tspan(env))
        env(action, position_mask)
        push!(y, cpu(env.signal))
        println(env.time_step)
    end

    return Episode(s, a, t, y)
end

function prepare_data(ep::Episode{S, Matrix{Float32}}, horizon::Int) where S
    s = S[]
    a = Vector{<: AbstractDesign}[]
    t = Vector{Float32}[]
    y = Matrix{Float32}[]
    s_ = Vector{S}[]

    n = horizon - 1
    for i in 1:(length(ep)-n-1) # added -1 to support next state
        boundary = i + n
        push!(s, ep.s[i])
        push!(a, ep.a[i:boundary])
        push!(t, flatten_repeated_last_dim(hcat(ep.t[i:boundary]...)))
        push!(s_, ep.s[i+1:i+horizon])

        signal = cat(ep.y[i:boundary]..., dims = 3)
        signal = permutedims(flatten_repeated_last_dim(permutedims(signal, (2, 1, 3))))
        push!(y, signal)
    end

    return s, a, t, y, s_
end

function prepare_data(eps::Vector{Episode{S, Y}}, horizon::Int) where {S, Y}
    return vcat.(prepare_data.(eps, horizon)...)
end

function FileIO.save(episode::Episode, path::String)
    BSON.bson(path, s = episode.s, a = episode.a, t = episode.t, y = episode.y)
end

function Episode(;path::String)
    file = BSON.load(path)
    s = identity.(file[:s])
    a = identity.(file[:a])
    t = identity.(file[:t])
    y = identity.(file[:y])
    return Episode(s, a, t, y)
end