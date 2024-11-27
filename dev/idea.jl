function SolverCore.solve!(
    solver::R2NSolver{T, V},
    nlp::AbstractNLPModel{T, V},
    stats::GenericExecutionStats{T, V};
    callback = (args...) -> nothing,
    x::V = nlp.meta.x0,
    atol::T = √eps(T),
    rtol::T = √eps(T),
    η1 = eps(T)^(1 / 4),
    η2 = T(0.95),
    γ1 = T(1 / 2),
    γ2 = 1 / γ1,
    σmin = zero(T),
    max_time::Float64 = 30.0,
    max_eval::Int = -1,
    max_iter::Int = typemax(Int),
    verbose::Int = 0,
    subsolver_verbose::Int = 0,
    non_mono_size = 1,
) where {T, V}
    unconstrained(nlp) || error("R2N should only be called on unconstrained problems.")

    reset!(stats)
    start_time = time()
    set_time!(stats, 0.0)

    x = solver.x .= x
    ∇fk = solver.gx    # k-1 gradient
    ∇ft = solver.gt    # current gradient
    s = solver.s
    B = solver.B
    ck = solver.cx
    σk = solver.σ
    cgtol = one(T)     # Must be ≤ 1.0

    reset!(B)

    set_iter!(stats, 0)
    set_objective!(stats, obj(nlp, x))

    grad!(nlp, x, ∇fk)
    norm_∇fk = norm(∇fk)
    set_dual_residual!(stats, norm_∇fk)

    σk = 2^round(log2(norm_∇fk + 1))
    n = nlp.meta.nvar
    # Stopping criterion:
    ϵ = atol + rtol * norm_∇fk
    optimal = norm_∇fk ≤ ϵ

    if optimal
        @info("Optimal point found at initial point")
        @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
        @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk σk
    end
    if verbose > 0 && mod(stats.iter, verbose) == 0
        @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
        infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk σk
    end

    set_status!(
        stats,
        get_status(
            nlp,
            elapsed_time = stats.elapsed_time,
            optimal = optimal,
            max_eval = max_eval,
            iter = stats.iter,
            max_iter = max_iter,
            max_time = max_time,
        ),
    )

    solver.σ = σk
    callback(nlp, solver, stats)
    σk = solver.σ

    done = stats.status != :unknown

    # Preallocate temporary vectors to avoid allocations
    tmp_vec = similar(x)       # For storing -∇fk and other intermediate results
    tmp_vec2 = similar(x)      # For gradient difference in L-BFGS update

    while !done
        cgtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * cgtol))

        # Avoid allocation for -∇fk by using a preallocated tmp_vec
        tmp_vec .= ∇fk
        tmp_vec .*= -1          # tmp_vec = -∇fk

        subsolve!(solver, s, B, tmp_vec, 0.0, cgtol, n, σk, subsolver_verbose)

        # Compute slope without allocations
        slope = dot(s, ∇fk)

        # Compute curv without allocations
        tmp_vec .= σk .* s      # tmp_vec = σk * s
        tmp_vec .+= ∇fk         # tmp_vec = ∇fk + σk * s
        curv = -dot(s, tmp_vec)
        ΔTk = -slope - curv / 2

        # Update ck without allocations
        ck .= x
        ck .+= s                # ck = x + s

        fck = obj(nlp, ck)
        if fck == -Inf
            set_status!(stats, :unbounded)
            break
        end

        if non_mono_size > 1  # Non-monotone behavior
            k = mod(stats.iter, non_mono_size) + 1
            solver.obj_vec[k] = stats.objective
            fck_max = maximum(solver.obj_vec)
            ρk = (fck_max - fck) / (fck_max - fck + ΔTk)
        else
            ρk = (stats.objective - fck) / ΔTk
        end

        # Update regularization parameters
        if ρk >= η2
            σk = max(σmin, γ1 * σk)
        elseif ρk < η1
            σk = σk * γ2
        end

        # Acceptance of the new candidate
        if ρk >= η1
            x .= ck
            # Update L-BFGS
            grad!(nlp, x, ∇ft)

            # Compute gradient difference without allocations
            tmp_vec2 .= ∇ft
            tmp_vec2 .-= ∇fk    # tmp_vec2 = ∇ft - ∇fk

            push!(B, s, tmp_vec2)
            set_objective!(stats, fck)

            # Copy ∇ft to ∇fk without allocations
            ∇fk .= ∇ft
            norm_∇fk = norm(∇fk)
        end

        set_iter!(stats, stats.iter + 1)
        set_time!(stats, time() - start_time)
        set_dual_residual!(stats, norm_∇fk)
        optimal = norm_∇fk ≤ ϵ

        if verbose > 0 && mod(stats.iter, verbose) == 0
            @info infoline
            infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk σk
        end

        set_status!(
            stats,
            get_status(
                nlp,
                elapsed_time = stats.elapsed_time,
                optimal = optimal,
                max_eval = max_eval,
                iter = stats.iter,
                max_iter = max_iter,
                max_time = max_time,
            ),
        )
        solver.σ = σk
        callback(nlp, solver, stats)
        σk = solver.σ

        done = stats.status != :unknown
    end

    set_solution!(stats, x)
    return stats
end

function subsolve!(R2N::R2NSolver, s, B, ∇f, atol, cgtol, n, σ, subsolver_verbose)
    if R2N.subsolver_type isa KrylovSolver
        Krylov.solve!(
            R2N.subsolver_type,
            (B + σ * I(n)),
            ∇f,
            atol = atol,
            rtol = cgtol,
            itmax = max(2 * n, 50),
            verbose = subsolver_verbose,
        )
        s .= R2N.subsolver_type.x
        # stats = R2N.subsolver_type.stats
    elseif R2N.subsolver_type isa ShiftedLBFGSSolver
        solve_shifted_system!(s, B, ∇f, σ)
    else
        error("Unsupported subsolver type")
    end
end