using Pkg
Pkg.activate(".")
println("Activated")

using DifferentialEquations
using Turing
using Distributions
using StaticArrays
using ForwardDiff
using Preferences
using StatsPlots
using Random

Random.seed!(02072022)


# set_preferences!(ForwardDiff, "nansafe_mode" => false)

println("Running functions")
include("model.jl")

@model function fit_cucker_smaile(data, cucker_smaile_problem, problem_p, global_p)
    β ~ Uniform(0.0, 1.0)
    K ~ InverseGamma(2.0, 3.0)
    var ~ InverseGamma(2.0, 3.0)
    (alg, save_every) = global_p
    (N, _, _) = problem_p
    new_problem_p = (N, K, β)

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



    prob = remake(cucker_smaile_problem, p=new_problem_p, u0=convert(Matrix{typeof(var)}, cucker_smaile_problem.u0))
    sol = solve(prob, alg, saveat=save_every)
    predicted = vec(sol)
    data .~ Turing.Normal.(predicted, var)
end


function main()
    # SETUP
    # Agent state is (xᵢ(t), vᵢ(t)) ∈ ℝᴺ × ℝᴺ
    ρ = 1.0 # Circle radius
    N = 40 # Number of agents

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

    u0 = rand(Uniform(-1.0, 1.0), 4, N)

    β = 0.3
    K = 0.9
    p = (N, K, β)
    tspan = (0.0, 5.0)

    alg = Rodas5()
    save_every = 0.1
    global_p = (alg, save_every)

    # CONSTRUCT PROBLEM AND SOLVE
    prob = ODEProblem(cuckersmale!, u0, tspan, p)
    sol = solve(prob, alg, saveat=save_every)

    # PLOT SOLUTION
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


    sol_data = vec(sol)
    sol_data .+= rand(Normal(0, 0.05), length(sol_data))
    model = fit_cucker_smaile(sol_data, prob, p, global_p)
    chain = sample(model, NUTS(), 1000)
    # chain = sample(model, NUTS(), MCMCThreads(), 1000, 4)
    # num_chains = 4
    # chains = mapreduce(c -> sample(model, NUTS(), 3000), chainscat, 1:num_chains)
end

println("Running main")
ch = main()

plot(ch)