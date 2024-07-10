
"""
    kramers_kronig(ωs, zs; tail = false) => ys
Calculate the Kramers-Kronig transformation of a given function such that `ys + im * zs` is
a retarded function (i.e., analytic in the upper half of the complex plane).

y(ω) = 1/π ∫dω' z(ω') / (ω' - ω)

# Input
- `ωs`: Grid of frequencies on which the function is defined.
- `zs`: Imaginary part of a retarded function.

# Output
- `ys`: Real part computed by the Kramers-Kronig transformation.
"""
function kramers_kronig(ωs, zs; tail = false)
    ys = zeros(length(ωs))

    # Linearly fit y in (ωs[j] - dω/2, ωs[j] + dω/2) and integrate.
    for j in 1:length(ωs)
        # Linear fit z = aω + b
        # Compute slope using the left and right points if available.
        if j == 1
            a = (zs[j+1] - zs[j]) / (ωs[j+1] - ωs[j])
        elseif j == length(ωs)
            a = (zs[j] - zs[j-1]) / (ωs[j] - ωs[j-1])
        else
            a = (zs[j+1] - zs[j-1]) / (ωs[j+1] - ωs[j-1])
        end
        b = zs[j] - a * ωs[j]

        if j == 1
            ωL = ωs[j] - (ωs[j+1] - ωs[j]) / 2
        else
            ωL = (ωs[j] + ωs[j-1]) / 2
        end

        if j == length(ωs)
            ωR = ωs[j] + (ωs[j] - ωs[j-1]) / 2
        else
            ωR = (ωs[j] + ωs[j+1]) / 2
        end

        for i in eachindex(ωs)
            ys[i] += a * (ωR - ωL)
            ys[i] += (a * ωs[i] + b) * log(abs((ωR - ωs[i]) / (ωL - ωs[i])))
        end
    end
    ys ./= π

    if tail
        dω = ωs[2] - ωs[1]
        for (i, ω) in enumerate(ωs)
            # 1 / √ω extrapolation
            ωL = ωs[1] - dω/2
            ωR = ωs[end] + dω/2
            yL = sqrt(-ωs[1]) * zs[1]
            yR = sqrt(ωs[end]) * zs[end]

            if abs(ω) < sqrt(eps(ω))
                ys[i] += -yL * 2 / sqrt(-ωL) / π
                ys[i] += yR * 2 / sqrt(ωR) / π
            elseif ω > 0
                ys[i] += yL * (2 * atan(sqrt(-ωL / ω)) - π) / sqrt(ω) / π
                ys[i] += yR * 2 * atanh(sqrt(ω / ωR)) / sqrt(ω) / π

            else  # ω < 0
                ys[i] += -yL * 2 * atanh(sqrt(ω / ωL)) / sqrt(-ω) / π
                ys[i] += yR * 2 * atan(sqrt(-ω / ωR)) / sqrt(-ω) / π

            end
        end
    end

    ys
end
