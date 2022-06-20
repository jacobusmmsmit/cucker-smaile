using StaticArrays
using ForwardDiff

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