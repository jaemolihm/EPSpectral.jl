@inline function occ_fermion(e, T)
    if T > sqrt(eps(eltype(T)))
        return 1 / (exp(e / T) + 1)
    elseif T >= 0
        return (1 - sign(e)) / 2
    else
        throw(ArgumentError("Temperature must be positive"))
    end
end

@inline function docc_fermion(e, T)
    if T > sqrt(eps(eltype(T)))
        z = exp(e / T)
        return - z / (z + 1)^2 / T
    elseif T >= 0
        return e == 0 ? Inf : zero(e)
    else
        throw(ArgumentError("Temperature must be positive"))
    end
end


@inline function occ_boson(e, T)
    if T > sqrt(eps(eltype(T)))
        return e == 0 ? zero(e) : 1 / expm1(e / T)
    elseif T >= 0
        return zero(e)
    else
        throw(ArgumentError("Temperature must be positive"))
    end
end

@inline function delta_gaussian(x :: T, η) where T
    exp(-(x / η)^2) / sqrt(T(π)) / η
end

@inline function delta_lorentzian(x :: T, η) where T
    η / T(π) / (x^2 + η^2)
end
