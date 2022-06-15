using DifferentialEquations
using Turing
using Distributions
using StatsPlots
using StaticArrays
using BenchmarkTools
using ForwardDiff

include("model.jl")

dart = Shape([(0.0, 0.0), (0.5, 1.0), (1.0, 0.0), (0.5, 0.5)])

@model function fit_cucker_smaile(data, cucker_smaile_problem, problem_p, global_p)
    β ~ Uniform(0.0, 1.0)
    K ~ InverseGamma(2.0, 3.0)
    var ~ InverseGamma(2.0, 3.0)

    (alg, save_every) = global_p
    (N, _, _) = problem_p
    new_problem_p = (N, K, β)

    # Debugging: (can be improved with display or show or smth)
    # inference_parameter = β
    # if typeof(inference_parameter) <: ForwardDiff.Dual
    #     println("Val: $(inference_parameter.value), Derivative: $(inference_parameter.partials)")
    # else
    #     println("Val: $inference_parameter")
    # end

    # inference_parameter = K
    # if typeof(inference_parameter) <: ForwardDiff.Dual
    #     println("Val: $(inference_parameter.value), Derivative: $(inference_parameter.partials)")
    # else
    #     println("Val: $inference_parameter")
    # end

    # @show convert(Matrix{typeof(var)}, cucker_smaile_problem.u0)
    # @show new_problem_p
    prob = remake(cucker_smaile_problem, p=new_problem_p, u0=convert(Matrix{typeof(var)}, cucker_smaile_problem.u0)) # , isinplace=true
    sol = solve(prob, alg, saveat=save_every)
    if sol.retcode != :Success
        throw(ErrorException("Unsuccessful with parameters: K = $(K), β = $(β)"))
    else
        println("nice meme")
    end
    predicted = vec(sol)
    # @show sizeof(predicted)
    # @show sizeof(data)
    data .~ Turing.Normal.(predicted, var)
end


function main()
    # SETUP
    N = 30 # Number of agents
    # Agent state is (xᵢ(t), vᵢ(t)) ∈ ℝᵈ × ℝᵈ
    u0 = rand(Uniform(-1.0, 1.0), 4, N)

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

    β = 0.3
    K = 0.9
    p = (N, K, β)
    # du = similar(u0)
    tspan = (0.0, 5.0)

    alg = RK4()
    save_every = 0.1
    global_p = (alg, save_every)

    # CONSTRUCT PROBLEM AND SOLVE
    prob = ODEProblem(cuckersmale!, u0, tspan, p)
    sol = solve(prob, alg, saveat=save_every)
    # sol_data = vec(sol)

    # model = fit_cucker_smaile(sol_data, prob, p, global_p)

    # sampling_algorithm = NUTS()
    # n_samples = 30
    # n_chains = 8
    # println("Running inference with $sampling_algorithm for $n_samples iterations on $n_chains independent chains: ")
    # chain = sample(model, sampling_algorithm, n_samples)
    # plot(chain)

    xvals = (sol(time)[1, :] for time in sol.t)
    yvals = (sol(time)[2, :] for time in sol.t)
    xrange = (minimum((minimum(minimum.(xvals)), 0)), maximum((maximum(maximum.(xvals)), 1)))
    yrange = (minimum((minimum(minimum.(yvals)), 0)), maximum((maximum(maximum.(yvals)), 1)))

    anim = @animate for t in tspan[1]:0.1:tspan[2]
        plot1 = plot(
            xlims=xrange,
            ylims=yrange,
            aspect_ratio=1.0,
            legend=:none,)
        for i in 1:N
            point = (sol(t)[1, i], sol(t)[2, i])
            plot!(
                plot1,
                point,
                msw=1.5,
                markersize=10,
                # marker=dart,
                mc="#4682B4",
                st=:scatter,
                msc=:black,
            )
        end
    end
    gif(anim, "gifs/circle_$N.gif", fps=60)
end

main()