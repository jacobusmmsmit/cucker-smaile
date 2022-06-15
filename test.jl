using DifferentialEquations
using StaticArrays
using Test
using StatsPlots

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

    β = 0.7
    K = 0.9
    p = (N, K, β)
    # du = similar(u0)
    tspan = (0.0, 2000.0)
    save_every = 0.1
    ntimes = Int(tspan[2] / save_every + 1)
    alg = RK4()

    # CONSTRUCT PROBLEM AND SOLVE
    prob = ODEProblem(cuckersmale!, u0, tspan, p)
    sol = solve(prob, alg, saveat=save_every)
    data = zeros(4, 30, ntimes)
    for (k, t) in enumerate(sol.t)
        data[:, :, k] .= sol(t)
    end
    data
    # test if every subsequent absolute value difference is roughly equal to 0 or greater than zero
    diff_data = zeros(4, 30, ntimes - 1)
    for k in 2:ntimes
        diff_data[:, :, k-1] = data[:, :, k] - data[:, :, k-1]
    end
    for k in 2:ntimes-1
        println(diff_data[:, :, k-1] .<= diff_data[:, :, k])
        # || isapprox.(diff_data[:, :, k], Ref(0); atol=0.0001)
    end
end


main()