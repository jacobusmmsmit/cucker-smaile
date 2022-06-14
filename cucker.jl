using DifferentialEquations
using Turing
using Distributions
using Plots


a(r, K, β) = K / ((1 + r^2)^β) # Communication kernel
twonorm(x) = sqrt(sum(abs2, x))

dart = Shape([(0.0, 0.0), (0.5, 1.0), (1.0, 0.0), (0.5, 0.5)])

function cuckersmale!(du, u, p, t)
    N, K, β = p

    for i in 1:N
        du[3, i] = zero(eltype(u))
        du[4, i] = zero(eltype(u))
        du[1, i] = u[3, i]
        du[2, i] = u[4, i]
    end

    for i in 1:N
        xi = (u[1, i], u[2, i])
        vi = (u[3, i], u[4, i])
        totx, toty = zero(eltype(u)), zero(eltype(u))
        for j in 1:N
            xj = (u[1, j], u[2, j])
            vj = (u[3, j], u[4, j])
            xdiff = xj .- xi
            vdiff = vj .- vi
            x = a(twonorm(xdiff), K, β) .* vdiff
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

    # CONSTRUCT PROBLEM AND SOLVE
    prob = ODEProblem(cuckersmale!, u0, tspan, p)
    sol = solve(prob)

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