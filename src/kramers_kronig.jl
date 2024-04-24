
function kramers_kronig(ωs, ys; tail = true)
    dω = ωs[2] - ωs[1]
    ys_out = zeros(length(ωs))

    for i in eachindex(ωs), j in eachindex(ωs)
        if i == j
            if i == 1
                ys_out[i] += ys[i+1] - ys[i]
            elseif i == length(ωs)
                ys_out[i] += ys[i] - ys[i-1]
            else
                ys_out[i] += (ys[i+1] - ys[i-1]) / 2
            end
        else
            ys_out[i] += ys[j] / (ωs[j] - ωs[i])
        end
    end
    ys_out .*= dω / π

    # for (i, ω) in enumerate(ωs)
    #     if abs(ω) < sqrt(eps(ω))
    #         # 1 / ω extrapolation at ω > ωs[end]
    #         ys_out[i] -= -ωs[end] * ys[end] / (ωs[end] + dω/2) / π

    #         # 1 / ω extrapolation at ω < ωs[1]
    #         ys_out[i] += -ωs[1] * ys[1] / (ωs[1] - dω/2) / π
    #     else
    #         # 1 / ω extrapolation at ω > ωs[end]
    #         ys_out[i] -= ωs[end] * ys[end] / ω * log1p(- ω / (ωs[end] + dω/2)) / π

    #         # 1 / ω extrapolation at ω < ωs[1]
    #         ys_out[i] += ωs[1] * ys[1] / ω * log1p(- ω / (ωs[1] - dω/2)) / π
    #     end
    # end

    if tail
        for (i, ω) in enumerate(ωs)
            # 1 / √ω extrapolation
            ωL = ωs[1] - dω/2
            ωR = ωs[end] + dω/2
            yL = sqrt(-ωs[1]) * ys[1]
            yR = sqrt(ωs[end]) * ys[end]

            if abs(ω) < sqrt(eps(ω))
                ys_out[i] += -yL * 2 / sqrt(-ωL) / π
                ys_out[i] += yR * 2 / sqrt(ωR) / π
            elseif ω > 0
                ys_out[i] += yL * (2 * atan(sqrt(-ωL / ω)) - π) / sqrt(ω) / π
                ys_out[i] += yR * 2 * atanh(sqrt(ω / ωR)) / sqrt(ω) / π

            else  # ω < 0
                ys_out[i] += -yL * 2 * atanh(sqrt(ω / ωL)) / sqrt(-ω) / π
                ys_out[i] += yR * 2 * atan(sqrt(-ω / ωR)) / sqrt(-ω) / π

            end
        end
    end

    ys_out
end
