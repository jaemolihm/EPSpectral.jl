# Fermi-Dirac and Bose-Einstein occupations, called inside the Fan-Migdal element function
# (`fm_term`) that a broadcast reduction runs on the host and on the device.
# ElectronPhonon exports functions of the same names, but its `occ_fermion` takes `occ_type` as a
# runtime keyword, whose branch chain and string-interpolating `throw` do not compile in a device
# kernel. These local copies are therefore not exported, and EPSpectral imports ElectronPhonon names
# explicitly so that EP's versions are not in scope here.
# TODO: upstream a device-compilable form to ElectronPhonon and delete these copies.

@inline function occ_fermion(e, T)
    if T > sqrt(eps(eltype(T)))
        return 1 / (exp(e / T) + 1)
    elseif T >= 0
        return (1 - sign(e)) / 2
    else
        throw(ArgumentError("Temperature must be positive"))
    end
end

@inline function occ_boson(e, T :: FT) where {FT}
    if T > sqrt(eps(FT))
        return e == 0 ? FT(-1/2) : 1 / expm1(e / T)
    elseif T >= 0
        if e > 0
            return FT(0)
        elseif e == 0
            return FT(-1/2)
        else
            return FT(-1)
        end
    else
        throw(ArgumentError("Temperature must be positive"))
    end
end
