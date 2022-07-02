using DifferentialEquations
using StaticArrays
using Test
using StatsPlots
using Measures

include("model.jl")

function main()
    # SETUP
    ρ = 1.0 # Circle radius
    N = 30 # Number of agents

    # Agent state is (xᵢ(t), vᵢ(t)) ∈ ℝᵈ × ℝᵈ
    angles = 0.0:2π/N:(2π-2π/N) # Angular position of agents
    positions = [SVector(ρ * sin(θ), ρ * cos(θ)) for θ in angles]
    velocities = .-positions
    u0 = zeros(4, N)
    for (i, (p, v)) in enumerate(zip(positions, velocities))
        u0[1, i] = p[1]
        u0[2, i] = p[2]
        u0[3, i] = v[1]
        u0[4, i] = v[2]
    end

    u0 = rand(4, N)

    β = 0.3
    K = 0.9
    p = (N, K, β)
    # du = similar(u0)
    tspan = (0.0, 10.0)
    save_every = 0.005
    ntimes = Int(tspan[2] / save_every + 1)
    alg = RK4()

    # CONSTRUCT PROBLEM AND SOLVE
    prob = ODEProblem(cuckersmale!, u0, tspan, p)
    sol = solve(prob, alg, saveat=save_every)
    data = zeros(4, 30, ntimes)
    for (k, t) in enumerate(sol.t)
        data[:, :, k] .= sol(t)
    end
    
    # p3 = plot(title = "Position of Particles")
    # for i in 1:N
    #     plot3d!(sol.t, data[1, i, :], data[2, i, :])
    # end
    # plot!(xlabel = "Time", ylabel = "x-position", zlabel = "y-position")

    p3 = plot(title = "x-position of Particles")
    for i in 1:N
        plot!(sol.t, data[1, i, :], label = "")
    end
    plot!(xlabel = "Time", ylabel = "x-position", xlims = [10^-2, 10])
    
    
    # data[1:2, 1, 1:10]





    # test if every subsequent absolute value difference is roughly equal to 0 or greater than zero
    # diff_final = zeros(4, 30, ntimes - 1)
    # norm_diff_final = zeros(2, 30, ntimes - 1)
    # norm_ratio = zeros(2, 30, ntimes - 1)
    # total_norm_ratio_p = zeros(ntimes - 1)
    # total_norm_ratio_v = zeros(ntimes - 1)
    # total_diff_final_p = zeros(ntimes - 1)
    # total_diff_final_v = zeros(ntimes - 1)
    # for k in 1:ntimes-1
    #     diff_final[:, :, k] = data[:, :, k] - data[:, :, end]
    #     for j in 1:N
    #         @views norm_diff_final[1, j, k] = twonorm(diff_final[1:2, j, k]) / N
    #         @views norm_diff_final[2, j, k] = twonorm(diff_final[3:4, j, k]) / N
    #     end
    # end
    # for k in 2:ntimes-1
    #     for j in 1:N
    #         @views norm_ratio[1, j, k-1] = norm_diff_final[1, j, k] / norm_diff_final[1, j, k-1]
    #         @views norm_ratio[2, j, k-1] = norm_diff_final[2, j, k] / norm_diff_final[2, j, k-1]
    #     end

    # end
    # for k in 1:ntimes-1
    #     total_norm_ratio_p[k] = sum(norm_ratio[1, :, k])
    #     total_norm_ratio_v[k] = sum(norm_ratio[2, :, k])
    #     total_diff_final_p[k] = sum(norm_diff_final[1, :, k])
    #     total_diff_final_v[k] = sum(norm_diff_final[2, :, k])
    # end
    # p1 = plot()
    # plot!(sol.t[1:end-1], total_diff_final_p, label="Position")
    # plot!(sol.t[1:end-1], total_diff_final_v, label="Velocity")
    # plot!(
    #     xlabel="Time (t)",
    #     ylabel="Average Distance to Final Value",
    #     margin=3mm,
    #     # xaxis = :log,
    #     # xlims = (save_every, tspan[2]),
    # )
    # p2 = plot(p1, xlabel="Log Time (t)", xlims=(save_every, tspan[2]), xaxis=:log, xticks=[10^-2, 10^-1, 10^0])
    # plot(p1, p2, layout=(2, 1), size=(500, 750))
    # p2 = plot(title = "Difference in 2-norm from Current to Final Value")
    # plot!(sol.t[1:end-2], total_norm_ratio_p[1:end-1], label = "Position")
    # plot!(sol.t[1:end-2], total_norm_ratio_v[1:end-1], label = "Velocity")
    # for k in 2:ntimes-1
    #     println(diff_data[:, :, k-1] .<= diff_data[:, :, k])
    #     # || isapprox.(diff_data[:, :, k], Ref(0); atol=0.0001)
    # end
end


main()