using Waves, Flux, CairoMakie, BSON, Statistics

colors = [  :red, :blue, :green, :orange, :purple, :cyan, :magenta, :yellow, :brown, :pink,
                :lime, :teal, :violet, :gold, :indigo, :olive, :navy, :coral, :turquoise, :salmon ]

function get_best_gbo(gbo_data, signal_index)
    steady_state_values = []
    for i in 1:length(gbo_data)
        for j in 1:length(gbo_data[i])
            push!(steady_state_values, gbo_data[i][j][:signal][signal_index, end])
        end
    end
    best_value = signal_index == 3 ? minimum(steady_state_values) : maximum(steady_state_values)
    return ones(Float32, length(gbo_data[1][1][:signal][signal_index, :])) * best_value
end

function get_best_gbo_single(gbo_data::Vector{Dict{Symbol, Any}}, signal_index)
    return get_best_gbo([gbo_data], signal_index)
end

function multiple_mpc_rendering(data_array, data_description, title, output_path, signal_index, gbo_ss=nothing)
    fig = Figure(;size = (2400, 1800))

    max_bound = 0
    max_energy = 0
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        max_energy = max(max_energy, maximum(vcat(vecs...)))
        upper_bound = maximum(mean(vecs) .+ 0.5 * sqrt.(var(vecs)))
        max_bound = max(max_bound, upper_bound)
    end

    ax_master_array = []
    ax_last_array = []
    average_array = []
    upper_bound_array = []
    lower_bound_array = []
    ax_average_array = []

    for i in 1:length(data_array) 
        ax_arr = []
        for j in 1:length(data_array[i])
            push!(ax_arr, Axis(fig[j, 2*i-1], aspect = 1.0, title = "$(String(colors[j]))", xlabel = "Space (m)", ylabel = "Space (m)"))
        end
        ax_last = Axis(fig[1:div(length(data_array[i]), 2)  , 2*i], title = "$title ($(data_description[i]))", xlabel = "Time (s)", ylabel = "Energy")
        xlims!(ax_last, t[1], round(t[end], digits=1))
        ylims!(ax_last, 0.0, max_energy * 1.20)
        
        push!(ax_master_array, ax_arr)
        push!(ax_last_array, ax_last)

        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        push!(average_array, mean(vecs))
        push!(upper_bound_array, average_array[end] .+ sqrt.(var(vecs)))
        push!(lower_bound_array, average_array[end] .- sqrt.(var(vecs)))

        ax_average = Axis(fig[(div(length(data_array[i]), 2) + 1):length(data_array[i]), 2*i], title = "Mean ± 0.5 * Standard Deviation", xlabel = "Time (s)", ylabel = "Energy")
        xlims!(ax_average, t[1], round(t[end], digits=1))
        ylims!(ax_average, 0.0, max_bound * 1.20)
        push!(ax_average_array, ax_average)
    end

    # ax_average = Axis(fig[(length(data_array[1]) + 1):2*length(data_array[1]), 1:(2*length(data_array))], title = "Mean ± 0.5 * Standard Deviation", xlabel = "Time (s)", ylabel = "Energy")
    # xlims!(ax_average, t[1], round(t[end], digits=1))
    # ylims!(ax_average, 0.0, max_bound * 1.20)

    # Bounding box coordinates
    x1, y1, x2, y2 = 0, 0, 15, 15

    CairoMakie.record(fig, output_path, axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
        println(i)
        # empty!(ax_average)
        for k in 1:length(data_array)
            for j in 1:length(data_array[k])
                empty!(ax_master_array[k][j])
                heatmap!(ax_master_array[k][j], dim.x, dim.y, data_array[k][j][:x][i], colormap = :ice, colorrange = (0.0, 0.2))
                mesh!(ax_master_array[k][j], Waves.multi_design_interpolation(Vector{DesignInterpolator}(data_array[k][j][:interps]), tspan[i]))
                if signal_index != 3
                    lines!(ax_master_array[k][j], [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=2, linestyle=:dot)
                end
            end

            idx = findfirst(tspan[i] .<= t)[1]
            empty!(ax_last_array[k])
            for j in 1:length(data_array[k])
                lines!(ax_last_array[k], t[1:idx], data_array[k][j][:signal][signal_index, 1:idx], color=colors[j])
            end

            empty!(ax_average_array[k])
            band!(ax_average_array[k], t[1:idx], lower_bound_array[k][1:idx], upper_bound_array[k][1:idx]; color = (:cyan, 0.4))
            lines!(ax_average_array[k], t[1:idx], average_array[k][1:idx], color=:blue)
            if !isnothing(gbo_ss)
                lines!(ax_average_array[k], t, gbo_ss, color=:orange)
            end
            # band!(ax_average, t[1:idx], lower_bound_array[k][1:idx], upper_bound_array[k][1:idx]; color = (colors[k], 0.4))
            # lines!(ax_average, t[1:idx], average_array[k][1:idx], color=colors[k])
        end
    end
end

function create_average_axis(fig, fig_indices, data_array, data_description, title, signal_index, gbo_ss=nothing)
    max_bound = 0
    max_average = 0
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        upper_bound = maximum(mean(vecs) .+ 0.5*sqrt.(var(vecs)))
        max_bound = max(max_bound, upper_bound)
        max_average = max(max_average, maximum(mean(vecs)))
    end
    
    ax_average = Axis(fig[fig_indices[1],fig_indices[2]], title = title, xlabel = "Time (s)", ylabel = "Scattered Energy")
    xlims!(ax_average, t[1], round(t[end], digits=1))
    ylims!(ax_average, 0.0, length(data_array[1]) > 1 ? max_bound * 1.20 : max_average * 1.20)
    empty!(ax_average)

    average_array = []
    std_dev_array = []
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        average = mean(vecs)
        upper_bound = average .+ 0.5*sqrt.(var(vecs))
        lower_bound = average .- 0.5*sqrt.(var(vecs))

        if length(data_array[1]) > 1
            band!(ax_average, t, lower_bound, upper_bound; color = (colors[i], 0.4))
        end
        lines!(ax_average, t, average, label="$(data_description[i])"; color = colors[i])
        
        push!(average_array, average)
        push!(std_dev_array, sqrt.(var(vecs)))
    end
    if !isnothing(gbo_ss)
        lines!(ax_average, t, gbo_ss, label="GBO"; color = colors[length(data_array)+2], linewidth = 6, linestyle = :dash)
    end
    axislegend(ax_average, position = :lt)
    return average_array, std_dev_array
end

function single_row_mpc(data_array, data_description, title, output_path, signal_index)
    fig = Figure(;size = (2000, 1600), figure_padding = (30, 50, 30, 30))
    Label(fig[1, 1:length(data_array)], title, halign = :center, fontsize = 36)

    grid_array = []
    for i in 1:length(data_array)
        push!(grid_array, Axis(fig[2, i], aspect = 1.0, title = data_description[i], xlabel = "Space (m)", ylabel = "Space (m)"))
    end
    x1, y1, x2, y2 = 0, 0, 15, 15

    max_average = 0
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        max_average = max(max_average, maximum(mean(vecs)))
    end
    ax_title = signal_index == 3 ? "Scattered Energy in Entire Grid" : "Scattered Energy in Top Right Quadrant"
    ax = Axis(fig[3, 1:length(data_array)], title = "", xlabel = "Time (s)", ylabel = "Scattered Energy")
    xlims!(ax, t[1], round(t[end], digits=1))
    ylims!(ax, 0.0, max_average * 1.20)
    empty!(ax)

    CairoMakie.record(fig, output_path, axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
        println(i)
        empty!(ax)
        for j in 1:length(data_array)
            empty!(grid_array[j])
            heatmap!(grid_array[j], dim.x, dim.y, data_array[j][1][:x][i], colormap = :ice, colorrange = (0.0, 0.2))
            mesh!(grid_array[j], Waves.multi_design_interpolation(Vector{DesignInterpolator}(data_array[j][1][:interps]), tspan[i]))
            if signal_index != 3
                lines!(grid_array[j], [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=2, linestyle=:dot)
            end
            idx = findfirst(tspan[i] .<= t)[1]

            lines!(ax, t[1:idx], data_array[j][1][:signal][signal_index, 1:idx], label="$(data_description[j])"; color = colors[j])
            axislegend(ax, position = :lt)
        end
    end

end

function create_figures(data_array, data_description, title, output_path, signal_index, gbo_ss=nothing)
    fig = Figure(;size = (2000, 2000), figure_padding = (30, 50, 30, 30))
    Label(fig[1, 1:length(data_array)], title, halign = :center, fontsize = 24)
    
    max_energy = 0
    for i in 1:length(data_array)
        vecs = [data_array[i][j][:signal][signal_index, :] for j in 1:length(data_array[i])]
        max_energy = max(max_energy, maximum(vcat(vecs...)))
    end

    for i in 1:length(data_array)
        ax_last = Axis(fig[2, i], title = data_description[i], xlabel = "Time (s)", ylabel = "Energy")
        xlims!(ax_last, t[1], round(t[end], digits=1))
        ylims!(ax_last, 0.0, max_energy * 1.20)
        empty!(ax_last)
        for j in 1:length(data_array[i])
            lines!(ax_last, t, data_array[i][j][:signal][signal_index, :], color=colors[j])
        end
        
    end

    create_average_axis(fig, (3, 1:length(data_array)), data_array, data_description, title, signal_index, gbo_ss)
    save(output_path, fig)

    fig2 = Figure(;size = (2000, 1200), figure_padding = (30, 50, 30, 30))
    average, std_dev = create_average_axis(fig2, (1, 1), data_array, data_description, title, signal_index, gbo_ss)
    save(joinpath(dirname(output_path), "average_.png"), fig2)

    return average, std_dev
end

function create_frame_by_frame_figure(data, w, h, output_path, signal_index, title)
    fig = Figure(;size = (2600, 1200))
    x1, y1, x2, y2 = 0, 0, 15, 15
    frame_indices = collect(1:div(frames,w*h):frames)
    frame_number = 1

    for i in 1:w
        for j in 1:h
            t_value = round((frame_number - 1) / (w * h) * round(t[end], digits=2), digits=3)
            ax = Axis(fig[i, j], aspect = 1.0, title = "t = $t_value")
            heatmap!(ax, dim.x, dim.y, data[:x][frame_indices[frame_number]], colormap = :ice, colorrange = (0.0, 0.2))
            mesh!(ax, Waves.multi_design_interpolation(Vector{DesignInterpolator}(data[:interps]), tspan[frame_indices[frame_number]]))
            if signal_index != 3
                lines!(ax, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=8, linestyle=:dot)
            end
            frame_number += 1
        end
    end
    # Label(fig[0, :], title, fontsize = 40, tellheight = false, tellwidth = true)
    save(output_path, fig)
end

function render_row(row_data, data_description, output_path, title)
    fig = Figure(;size = (1100, 700), figure_padding = (30, 50, 30, 30))

    max_energy = maximum(vcat([row_data[i][:signal][13, :] for i in 1:length(row_data)]...))

    ax_grid = []
    for i in 1:length(row_data) 
        push!(ax_grid, Axis(fig[1, i], aspect = 1.0, title = data_description[i], xlabel = "Space (m)", ylabel = "Space (m)"))
    end
    ax = Axis(fig[2, 1:length(row_data)], title = title, xlabel = "Time (s)", ylabel = "Scattered Energy")
    xlims!(ax, t[1], round(t[end], digits=1))
    ylims!(ax, 0.0, max_energy * 1.20)

    x1, y1, x2, y2 = 0, 0, 15, 15

    CairoMakie.record(fig, output_path, axes(tspan, 1), framerate = Waves.FRAMES_PER_SECOND) do i
        println(i)
        empty!(ax)
        idx = findfirst(tspan[i] .<= t)[1]
        for k in 1:length(row_data)
            empty!(ax_grid[k])
            heatmap!(ax_grid[k], dim.x, dim.y, row_data[k][:x][i], colormap = :ice, colorrange = (0.0, 0.2))
            mesh!(ax_grid[k], Waves.multi_design_interpolation(Vector{DesignInterpolator}(row_data[k][:interps]), tspan[i]))
            lines!(ax_grid[k], [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=:red, linewidth=2, linestyle=:dot)

            lines!(ax, t[1:idx], row_data[k][:signal][13, 1:idx], label=data_description[k], color=colors[k])
        end
        axislegend(ax, position = :lt)
    end
end


# dataset_name = "pos_adjustment_masked_signals_M=1"
dataset_name = "full_adjustment_masked_signals_M=2"
DATA_PATH = "scratch/$dataset_name"
@time env = BSON.load(joinpath(DATA_PATH, "env.bson"))[:env]
dim = env.dim

env.actions = 200
t = build_tspan(0.0f0, env.dt, env.actions * env.integration_steps)
seconds = 40.0
frames = Int(round(Waves.FRAMES_PER_SECOND * seconds))
tspan = collect(range(t[1], t[end], frames))

my_theme = Theme(fontsize = 26)
set_theme!(my_theme)



# RENDER ROW EXAMPLE
# prefixes = ["mpc", "random"]
# data_description = ["AEM", "Random"]
# folder = "tutorial_focusing.mpc"
# row_index = 1
# @time focus = [BSON.load(joinpath(folder, "$(prefixes[i])_$(row_index).bson")) for i in 1:length(prefixes)]
# output_path = "mpc_row_example.mp4"
# title = "Focusing"

# render_row(focus, data_description, output_path, title)
# END EXAMPLE


# MULTIPLE MPC EXAMPLE
# index_array = vcat(1:3)
# folder = "tutorial_focusing.mpc"
# @time mpc_data = [BSON.load(joinpath(folder, "mpc_$i.bson")) for i in index_array]
# @time random_data = [BSON.load(joinpath(folder, "random_$i.bson")) for i in index_array]
# data_array = [mpc_data, random_data]
# data_description = ["AEM", "Random"]
# title = "Focusing"
# output_path = "multiple_mpc_example.mp4"
# focusing = true

# multiple_mpc_rendering(data_array, data_description, title, output_path, focusing ? 13 : 3)
# END EXAMPLE


# CREATE FIGURES EXAMPLE
# index_array = vcat(1:3)
# folder = "tutorial_focusing.mpc"
# @time mpc_data = [BSON.load(joinpath(folder, "mpc_$i.bson")) for i in index_array]
# @time random_data = [BSON.load(joinpath(folder, "random_$i.bson")) for i in index_array]
# data_array = [mpc_data, random_data]
# data_description = ["AEM", "Random"]
# title = "Focusing"
# output_path = "create_figures_example.png"
# focusing = true

# create_figures(data_array, data_description, title, output_path, focusing ? 13 : 3)
# END EXAMPLE


# FRAME BY FRAME EXAMPLE
@time mpc_data = [BSON.load(joinpath(folder, "mpc_$i.bson")) for i in index_array]
focusing = true

create_frame_by_frame_figure(mpc_data[1], 2, 6, "frameByFrame_example.png", focusing ? 13 : 3, "AEM, M=2, Full Adjustment")
# END EXAMPLE
