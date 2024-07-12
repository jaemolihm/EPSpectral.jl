# Holstein model for the electron-phonon interaction
# We use the atomic Hartree units (ħ = mₑ = e² = 4πε0 = 1).

struct HolsteinLatticeModel
    alat :: Float64
    g  :: Float64
    ω₀ :: Float64
    t  :: Float64
    μ  :: Float64
    T  :: Float64
end

Base.Broadcast.broadcastable(model::HolsteinLatticeModel) = Ref(model)

get_εk(k, model::HolsteinLatticeModel) = -2 * model.t * cos(k * model.alat)

get_vk(k, model::HolsteinLatticeModel) = 2 * model.t * model.alat * sin(k * model.alat)

get_eph_g(q, model :: HolsteinLatticeModel) = model.g
