
struct Kpoints{T}
    vectors :: Vector{T}
    weights :: Vector{Float64}
end

# Indexing and iteration interface
Base.length(k :: Kpoints) = length(k.vectors)
Base.getindex(k :: Kpoints, i::Int) = (k.vectors[i], k.weights[i])
Base.firstindex(k :: Kpoints) = 1
Base.lastindex(k :: Kpoints) = length(k)
Base.iterate(k :: Kpoints, i = 1) = i > length(k) ? nothing : (k[i], i + 1)

get_dimension(k :: Kpoints) = length(k.vectors[1])


"""
    polynomial_grid_1d(kmax, n, order = 1)
Generate a 1D grid with polynomial spacing in [-kmax, kmax].
For n = 2m - 1: ``k[i+m] = kmax * (i / (m - 1))^order``  (i = 1, ..., m - 1)
For n = 2m    : ``k[i+m] = kmax * ((i - 1/2) / (m - 1/2))^order`` (i = 1, ..., m)
"""
function polynomial_grid_1d(kmax, n :: Int, order :: Int = 1)
    if mod(n, 2) == 1
        m = div(n + 1, 2)
        ks_positive = @. kmax * ((1:(m-1)) ./ (m - 1))^order
        ks = vcat(.-reverse(ks_positive), 0, ks_positive)
    else
        m = div(n, 2)
        ks_positive = @. kmax * (((1:m) - 1/2) ./ (m - 1/2))^order
        ks = vcat(.-reverse(ks_positive), ks_positive)
    end

    weights_ = (ks[3:end] .- ks[1:end-2]) ./ 2
    weights = vcat(weights_[1], weights_, weights_[end])

    Kpoints(ks, weights)
end

"""
    polynomial_grid_2d(kmax, n, order = 1)
Generate a 2D grid with polynomial spacing in each directions inside the sphere with radius
kmax.
"""
function polynomial_grid_2d(kmax, n :: Int, order :: Int = 1)
    kpts_1d = polynomial_grid_1d(kmax, n, order)
    vectors_1d, weights_1d = kpts_1d.vectors, kpts_1d.weights

    vectors = vec(SVector.(collect(Iterators.product(vectors_1d, vectors_1d))))
    weights = vec(prod.(collect(Iterators.product(weights_1d, weights_1d))))

    # Filter out k points with |k| > kmax
    inds = norm.(vectors) .<= kmax
    Kpoints(vectors[inds], weights[inds])
end


"""
    polynomial_grid_3d(kmax, n, order = 1)
Generate a 3D grid with polynomial spacing in each directions inside the sphere with radius
kmax.
"""
function polynomial_grid_3d(kmax, n :: Int, order :: Int = 1)
    kpts_1d = polynomial_grid_1d(kmax, n, order)
    vectors_1d, weights_1d = kpts_1d.vectors, kpts_1d.weights

    vectors = vec(SVector.(collect(Iterators.product(vectors_1d, vectors_1d, vectors_1d))))
    weights = vec(prod.(collect(Iterators.product(weights_1d, weights_1d, weights_1d))))

    # Filter out k points with |k| > kmax
    inds = norm.(vectors) .<= kmax
    Kpoints(vectors[inds], weights[inds])
end


"""
    azimuthal_grid_3d(kmax, n, order)
Generate a 3D grid with azimuthal symmetry along the x axis.
"""
function azimuthal_grid_3d(kmax, nk, nθ, order = 1)
    # Radial grid: k² dk
    ks = @. kmax * ((1:nk) ./ nk)^order
    weights_k = ks.^2 .* vcat(
        ks[1] + ks[2] / 2,
        (ks[3:end] .- ks[1:end-2]) ./ 2,
        (ks[end] - ks[end-1]) / 2
    )

    # Angular grid: sinθ dθ = d(cosθ)
    # Compute weights from the trapezoidal rule
    cosθs = range(-1, 1, length=nθ)
    weights_θ = fill(2 / (nθ - 1), nθ)
    weights_θ[1] /= 2
    weights_θ[end] /= 2

    vectors = map(Iterators.product(ks, cosθs)) do (k, cosθ)
        sinθ = sqrt(1 - cosθ^2)
        SVector(k * cosθ, k * sinθ, 0)
    end |> vec

    # Multiply 2π for the azimuthal weight
    weights = vec(prod.(collect(Iterators.product(weights_k, weights_θ)))) .* 2π

    Kpoints(vectors, weights)
end
