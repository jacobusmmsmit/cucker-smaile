using Distributed
using DifferentialEquations
using Turing
using Distributions
using StaticArrays
using ForwardDiff
using Preferences
using StatsPlots

set_preferences!(ForwardDiff, "nansafe_mode" => true)


a(r, K, β) = K / ((1 + r^2)^β) # Communication kernel
twonorm(x) = sqrt(sum(abs2, x))

function cuckersmale!(du, u, p, t)
    N, K, β = p
    for i in 1:N
        xi = SVector(u[1, i], u[2, i])
        vi = SVector(u[3, i], u[4, i])
        totx, toty = zero(eltype(u)), zero(eltype(u))
        for j in 1:N
            xj = SVector(u[1, j], u[2, j])
            vj = SVector(u[3, j], u[4, j])
            x = a(twonorm(xj - xi), K, β) .* (vj - vi)
            totx += x[1]
            toty += x[2]
        end
        du[3, i] = totx / N
        du[4, i] = toty / N
        du[1, i] = u[3, i] + du[3, i]
        du[2, i] = u[4, i] + du[4, i]
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

    N = 2 # Number of agents
    u0 = rand(Uniform(-1.0, 1.0), 4, N)
    β = 0.3
    K = 0.9
    p = (N, K, β)
    tspan = (0.0, 5.0)

    alg = TRBDF2()
    save_every = 0.1
    global_p = (alg, save_every)

    # CONSTRUCT PROBLEM AND SOLVE
    prob = ODEProblem(cuckersmale!, u0, tspan, p)
    sol = solve(prob, alg, saveat=save_every)
    sol_data = vec(sol)
    sol_data .+= rand(Normal(0, 0.05), length(sol_data))
    model = fit_cucker_smaile(sol_data, prob, p, global_p)
    # chain = sample(model, NUTS(), 1000, 4)
    # chain = sample(model, NUTS(), 3000)
    num_chains = 4
    chains = mapreduce(c -> sample(model, NUTS(), 3000), chainscat, 1:num_chains)
end

ch = main()

plot(ch)