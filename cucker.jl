using DifferentialEquations
using Turing
using Distributions
using StatsPlots
using StaticArrays
using BenchmarkTools
using ForwardDiff

a(r, K, β) = K / ((1 + r^2)^β) # Communication kernel
twonorm(x) = sqrt(sum(abs2, x))

dart = Shape([(0.0, 0.0), (0.5, 1.0), (1.0, 0.0), (0.5, 0.5)])

function cuckersmale!(du, u, p, t)
    N, K, β = p

    # println(eltype(u))
    for i in 1:N
        du[3, i] = zero(eltype(u))
        du[4, i] = zero(eltype(u))
        du[1, i] = u[3, i]
        du[2, i] = u[4, i]
    end

    for i in 1:N
        xi = SVector(u[1, i], u[2, i])
        vi = SVector(u[3, i], u[4, i])
        totx, toty = zero(eltype(u)), zero(eltype(u))
        for j in 1:N
            xj = SVector(u[1, j], u[2, j])
            vj = SVector(u[3, j], u[4, j])
            xdiff = xj - xi
            vdiff = vj - vi
            x = a(twonorm(xdiff), K, β) .* vdiff
            # typeof(x) <: ForwardDiff.Dual && println("Val: $(x.value), Derivative: $(x.partials)")
            totx += x[1]
            toty += x[2]
        end
        du[3, i] += totx / N
        du[4, i] += toty / N
        du[1, i] += du[3, i]
        du[2, i] += du[4, i]
    end
    return nothing
end

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
        println("bruh moment")
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
    β = 0.3
    K = 0.9
    p = (N, K, β)
    # du = similar(u0)
    tspan = (0.0, 5.0)

    alg = TRBDF2()
    save_every = 0.1
    global_p = (alg, save_every)

    # CONSTRUCT PROBLEM AND SOLVE
    prob = ODEProblem(cuckersmale!, u0, tspan, p)
    sol = solve(prob, alg, saveat=save_every)
    sol_data = vec(sol)

    model = fit_cucker_smaile(sol_data, prob, p, global_p)

    sampling_algorithm = NUTS()
    n_samples = 30
    # n_chains = 8
    # println("Running inference with $sampling_algorithm for $n_samples iterations on $n_chains independent chains: ")
    chain = sample(model, sampling_algorithm, n_samples)
    plot(chain)

    # xvals = (sol(time)[1, :] for time in sol.t)
    # yvals = (sol(time)[2, :] for time in sol.t)
    # xrange = (minimum((minimum(minimum.(xvals)), 0)), maximum((maximum(maximum.(xvals)), 1)))
    # yrange = (minimum((minimum(minimum.(yvals)), 0)), maximum((maximum(maximum.(yvals)), 1)))

    # anim = @animate for t in tspan[1]:0.1:tspan[2]
    #     plot1 = plot(
    #         xlims=xrange,
    #         ylims=yrange,
    #         aspect_ratio=1.0,
    #         legend=:none,)
    #     for i in 1:N
    #         point = (sol(t)[1, i], sol(t)[2, i])
    #         plot!(
    #             plot1,
    #             point,
    #             msw=1.5,
    #             markersize=10,
    #             # marker=dart,
    #             mc="#4682B4",
    #             st=:scatter,
    #             msc=:black,
    #         )
    #     end
    # end
    # gif(anim, "gifs/first_out.gif", fps = 60)
end

main()