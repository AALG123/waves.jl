using Waves
using Flux
using ReinforcementLearning
using CairoMakie
using BSON
using Images: imresize
using Distributed
using ProgressMeter
using Base.Threads
include("../src/masks.jl")

function build_rectangular_grid(nx::Int, ny::Int, r::Float32)
    x = []

    for i in 1:nx
        for j in 1:ny
            push!(x, [(i-1) * 2 * r, (j-1) * 2 * r])
        end
    end

    x = hcat(x...)

    return x .- Flux.mean(x, dims = 2)
end

function build_rectangular_grid_design_space()
    pos = Matrix(build_rectangular_grid(5, 5, 1.0f0 + 0.1f0)')
    M = size(pos, 1)

    low = AdjustableRadiiScatterers(Cylinders(pos, fill(0.2f0, M), fill(3 * AIR, M)))
    high = AdjustableRadiiScatterers(Cylinders(pos, fill(1.0f0, M), fill(3 * AIR, M)))
    return DesignSpace(low, high)
end

# OPTIMIZATION: Parallel episode generation function
# This function generates multiple episodes simultaneously using threading
function generate_episodes_batch(env, policy, masks, batch_size::Int, start_idx::Int)
    # Pre-allocate array for better memory management
    episodes = Vector{Any}(undef, batch_size)
    
    # OPTIMIZATION: Use threading for parallel episode generation
    # Each thread works on a different episode simultaneously
    Threads.@threads for i in 1:batch_size
        # OPTIMIZATION: Create thread-local environment copy for thread safety
        # This prevents race conditions when multiple threads access the same environment
        local_env = deepcopy(env)
        episodes[i] = generate_episode!(policy, local_env, position_mask=masks)
    end
    
    return episodes
end

# OPTIMIZATION: Parallel saving function with batch processing
# Saves multiple episodes simultaneously to reduce I/O bottleneck
function save_episodes_batch(episodes, path, start_idx::Int)
    # OPTIMIZATION: Parallel file I/O using threading
    # Each thread saves a different episode file simultaneously
    Threads.@threads for i in 1:length(episodes)
        episode_idx = start_idx + i - 1
        save(episodes[i], joinpath(path, "episodes/episode$episode_idx.bson"))
    end
end

# OPTIMIZATION: Added informative startup messages
println("Starting optimized episode generation...")
println("Using $(Threads.nthreads()) threads")

Flux.device!(1)
DATA_PATH = "./scratch/"

dim = TwoDim(15.0f0, 700)
μ_low = [-10.0f0 0.0f0]
μ_high = [-10.0f0 0.0f0]
σ = [0.3f0]
a = [1.0f0]

function Base.rand(space::DesignSpace{NoDesign})
    return NoDesign()
end

M = 1
r = fill(1.0f0, M)
c = fill(AIR * 3, M)

low_pos = hcat(fill(-8.0f0, M), fill(-8.0f0, M))
high_pos = hcat(fill(8.0f0, M), fill(8.0f0, M))

low = AdjustablePositionScatterers(Cylinders(low_pos, r, c))
high = AdjustablePositionScatterers(Cylinders(high_pos, r, c))

design_space = DesignSpace(low, high)

masks = cat(create_patches(700, 350)..., dims = 3)

env = gpu(WaveEnv(dim; 
    design_space=design_space,
    source = RandomPosGaussianSource(build_grid(dim), μ_low, μ_high, σ, a, 1000.0f0),
    integration_steps = 100,
    actions = 200
    ))

policy = RandomDesignPolicy(action_space(env))

# OPTIMIZATION: Updated dataset name to reflect optimizations
name = "pos_adjustment_masked_signals_M=1_optimized"
path = mkpath(joinpath(DATA_PATH, name))
mkpath(joinpath(path, "episodes/"))
BSON.bson(joinpath(path, "env.bson"), env = cpu(env))

# OPTIMIZATION: Significantly increased episode count and added batch processing
TOTAL_EPISODES = 5000  # OPTIMIZATION: Increased from 500 to 5000 episodes (10x more data)
# OPTIMIZATION: Adaptive batch size based on available threads
# More threads = larger batches for better parallel efficiency
BATCH_SIZE = min(50, Threads.nthreads() * 8)  
# OPTIMIZATION: Regular progress saving to prevent data loss
SAVE_FREQUENCY = 100  # Save progress every 100 episodes

println("Generating $TOTAL_EPISODES episodes in batches of $BATCH_SIZE")
println("Progress will be saved every $SAVE_FREQUENCY episodes")

# OPTIMIZATION: Visual progress tracking for user feedback
progress = Progress(TOTAL_EPISODES, desc="Generating episodes: ")

# OPTIMIZATION: Main batch processing loop for efficient episode generation
for batch_start in 1:BATCH_SIZE:TOTAL_EPISODES
    batch_end = min(batch_start + BATCH_SIZE - 1, TOTAL_EPISODES)
    current_batch_size = batch_end - batch_start + 1
    
    # OPTIMIZATION: Generate episodes in parallel batches
    # This is much faster than sequential generation
    episodes = generate_episodes_batch(env, policy, masks, current_batch_size, batch_start)
    
    # OPTIMIZATION: Save episodes in parallel to reduce I/O time
    save_episodes_batch(episodes, path, batch_start)
    
    # OPTIMIZATION: Update progress bar for user feedback
    update!(progress, batch_end)
    
    # OPTIMIZATION: Periodic progress save and memory cleanup
    if batch_end % SAVE_FREQUENCY == 0 || batch_end == TOTAL_EPISODES
        println("\nCompleted $batch_end/$TOTAL_EPISODES episodes")
        
        # OPTIMIZATION: Force garbage collection to free memory
        # This prevents memory buildup during long-running jobs
        GC.gc()
        
        # OPTIMIZATION: Save checkpoint information for resumability
        # Allows restarting from last checkpoint if job fails
        checkpoint_info = Dict(
            "completed_episodes" => batch_end,
            "total_episodes" => TOTAL_EPISODES,
            "timestamp" => now(),
            "batch_size" => BATCH_SIZE
        )
        BSON.bson(joinpath(path, "checkpoint.bson"), checkpoint_info)
    end
end

# OPTIMIZATION: Final completion messages
println("\nEpisode generation completed!")
println("Generated $TOTAL_EPISODES episodes")
println("Dataset saved to: $path")

# OPTIMIZATION: Enhanced custom environment function with better naming
function custom_env()
    dim = TwoDim(15.0f0, 700)
    μ_low = [-10.0f0 0.0f0]
    μ_high = [-10.0f0 0.0f0]
    σ = [0.3f0]
    a = [1.0f0]

    # OPTIMIZATION: Increased M from 2 to 15 for more complex scenarios
    M = 15
    r = fill(1.0f0, M)
    c = fill(AIR * 3, M)

    low_pos = hcat(fill(-8.0f0, M), fill(-8.0f0, M))
    high_pos = hcat(fill(8.0f0, M), fill(8.0f0, M))

    low = AdjustablePositionScatterers(Cylinders(low_pos, r, c))
    high = AdjustablePositionScatterers(Cylinders(high_pos, r, c))

    design_space = DesignSpace(low, high)

    env = WaveEnv(dim; 
        design_space=design_space,
        source = RandomPosGaussianSource(build_grid(dim), μ_low, μ_high, σ, a, 1000.0f0),
        integration_steps = 100,
        actions = 200
        )

    DATA_PATH = "./scratch/"
    # OPTIMIZATION: Updated naming to reflect M=15 and optimization status
    name = "pos_adjustment_masked_M=15_optimized"
    path = mkpath(joinpath(DATA_PATH, name))
    BSON.bson(joinpath(path, "env_1.bson"), env = cpu(env))
    
    # OPTIMIZATION: Return both environment and path for potential further use
    return env, path
end
