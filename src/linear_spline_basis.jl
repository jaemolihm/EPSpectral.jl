struct LinearSplineBasis
    N :: Int
    xs :: Vector{Float64}
    LinearSplineBasis(xs) = new(length(xs) - 2, xs)
end

Base.Broadcast.broadcastable(basis :: LinearSplineBasis) = Ref(basis)

function grid_points(basis :: LinearSplineBasis)
    @view basis.xs[2:end-1]
end


# Basis evaluation
function (basis::LinearSplineBasis)(x, coeff)
    (; xs, N) = basis
    (length(coeff) == N) || throw(ArgumentError("length(coeff) != N"))

    if x <= xs[1] || x >= xs[end]
        return zero(x)
    else
        i = searchsortedlast(xs, x)
        val = zero(x)
        if i > 1
            val += coeff[i-1] * (xs[i+1] - x) / (xs[i+1] - xs[i])
        end
        if i <= N
            val += coeff[i] * (x - xs[i]) / (xs[i+1] - xs[i])
        end
        return val
    end
end

# Kramers-Kronig transformation

xlogx(x::Real) = iszero(x) ? zero(x) : x * log(abs(x))
xlogx(x::Complex) = x * log(x)

"""
    linear_spline_kramers_kronig(x, x1, x2, x3)
Kramers-Kronig transformation of a linear spline function at grid points `x1`, `x2`, `x3`
evaluated at `x`.
Evaluate ``Pval ∫ dy b(y; x1, x2, x3) / (y - x)``, where ``b(y; x1, x2, x3)`` is a
piecewise-linear function with ``b(x1) = b(x3) = 0`` and ``b(x2) = 1``.
"""
function linear_spline_kramers_kronig(x, x1, x2, x3)
    (
        + xlogx(x1 - x) / (x2 - x1)
        - xlogx(x2 - x) * (x3 - x1) / (x2 - x1) / (x3 - x2)
        + xlogx(x3 - x) / (x3 - x2)
    )
end

function get_Kramers_Kronig_map(basis :: LinearSplineBasis)
    (; N, xs) = basis
    KK_map = zeros(N, N)
    for j in 1:N, i in 1:N
        KK_map[i, j] += linear_spline_kramers_kronig(xs[i+1], xs[j], xs[j+1], xs[j+2])
    end
    KK_map
end

function apply_Kramers_Kronig(basis :: LinearSplineBasis, ys, x)
    (; N, xs) = basis
    mapreduce(+, 1:N) do i
        linear_spline_kramers_kronig(x, xs[i], xs[i+1], xs[i+2]) * ys[i]
    end
end

function get_Kramers_Kronig_map(xs_in :: Vector)
    N = length(xs_in)
    # Pad xs with equidistant spacing
    x0 = xs_in[1] - (xs_in[2] - xs_in[1])
    xN1 = xs_in[end] + (xs_in[end] - xs_in[end-1])
    xs = vcat(x0, xs_in, xN1)

    # KK_map = zeros(N, N)
    # for j in 1:N, i in 1:N
    #     KK_map[i, j] += linear_spline_kramers_kronig(xs[i+1], xs[j], xs[j+1], xs[j+2])
    # end
    ijs = collect(Iterators.product(1:N, 1:N))
    KK_map = tmap(ijs) do (i, j)
        linear_spline_kramers_kronig(xs[i+1], xs[j], xs[j+1], xs[j+2])
    end :: Matrix{Float64}
    KK_map
end
