using LinearAlgebra
using LinearOperators
using Krylov
using SolverCore

export R2NLS, R2NLSSolver

mutable struct R2NLSSolver{T, V, Sub <: KrylovSolver{T, T, V}} <: AbstractOptimizationSolver
    x::V
    ck::V
    Fk::V
    Jk::Matrix{T}
    σ::T
    s::V
    obj_vec::Vector{T}
    subsolver_type::Sub
    cgtol::T
end

function R2NLSSolver(
    nlp::AbstractNLSModel{T, V};
    non_mono_size = 1,
    subsolver_type::Union{Type{<:KrylovSolver}} = LSMRSolver,
) where {T, V}
    nvar = nlp.meta.nvar
    nres = nlp.meta.ncon
    x = V(undef, nvar)
    ck = V(undef, nvar)
    Fk = V(undef, nres)
    Jk = zeros(T, nres, nvar)
    s = V(undef, nvar)
    obj_vec = fill(typemin(T), non_mono_size)
    σ = zero(T)
    cgtol = one(T)
    subsolver = subsolver_type(nres + nvar)
    Sub = typeof(subsolver)
    return R2NLSSolver{T, V, Sub}(x, ck, Fk, Jk, σ, s, obj_vec, subsolver, cgtol)
end

function R2NLS(
    nlp::AbstractNLSModel{T, V};
    subsolver_type::Union{Type{<:KrylovSolver}} = LSMRSolver,
    non_mono_size = 1,
    kwargs...,
) where {T, V}
    solver = R2NLSSolver(nlp, non_mono_size=non_mono_size, subsolver_type=subsolver_type)
    return solve!(solver, nlp; non_mono_size=non_mono_size, kwargs...)
end

function SolverCore.solve!(
    solver::R2NLSSolver{T, V},
    nlp::AbstractNLSModel{T, V},
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
    unconstrained(nlp) || error("R2NLS should only be called on unconstrained nonlinear least squares problems.")
    if non_mono_size < 1
        error("non_mono_size must be greater than or equal to 1")
    end

    reset!(stats)
    start_time = time()
    set_time!(stats, 0.0)

    n = nlp.meta.nvar
    m = nlp.meta.ncon  # number of residuals

    x = solver.x .= x
    ck = solver.ck
    Fk = solver.Fk
    Jk = solver.Jk
    s = solver.s
    σk = solver.σ
    obj_vec = solver.obj_vec
    cgtol = solver.cgtol
    subsolver = solver.subsolver_type

    set_iter!(stats, 0)
    residual!(nlp, x, Fk)
    set_objective!(stats, 0.5 * dot(Fk, Fk))

    # Compute the Jacobian
    jacobian!(nlp, x, Jk)

    # Compute gradient gk = Jk' * Fk
    gk = Jk' * Fk
    norm_gk = norm(gk)
    set_dual_residual!(stats, norm_gk)

    σk = 2^round(log2(norm_gk + 1))

    # Stopping criterion:
    ϵ = atol + rtol * norm_gk
    optimal = norm_gk ≤ ϵ
    if optimal
        @info("Optimal point found at initial point")
        @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
        @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_gk σk
    end
    if verbose > 0 && mod(stats.iter, verbose) == 0
        @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "σ"
        infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_gk σk
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
    cgtol = max(rtol, min(T(0.1), √norm_gk, T(0.9) * cgtol))

    while !done
        # Subproblem solve:
        # Solve min_s (1/2)|| Fk + Jk s ||^2 + (1/2) σk ||s||^2

        sqrtσk = sqrt(σk)
        # Define the linear operator for A = [Jk; sqrtσk * I]
        function matvec(v)
            vcat(Jk * v, sqrtσk * v)
        end
        function rmatvec(u)
            u1 = view(u, 1:m)
            u2 = view(u, m+1:m+n)
            Jk' * u1 + sqrtσk * u2
        end
        A = LinearOperator{T}(m + n, n, matvec=matvec, rmatvec=rmatvec)

        b = -vcat(Fk, zeros(T, n))

        # Use LSMR to solve min_s || A * s - b ||
        Krylov.solve!(
            subsolver,
            A,
            b,
            atol = atol,
            rtol = cgtol,
            itmax = 2*n,
            verbose = subsolver_verbose,
        )
        s .= subsolver.x

        ck .= x .+ s
        residual!(nlp, ck, Fk)
        fck = 0.5 * dot(Fk, Fk)
        if fck == -Inf
            set_status!(stats, :unbounded)
            break
        end

        # Compute the predicted reduction ΔTk
        # ΔTk = - (1/2)*( || Fk + Jk s ||^2 + σk ||s||^2 - ||Fk||^2 )
        # For efficiency, compute ΔTk = - (0.5 * (norm(Fk + Jk * s)^2 + σk * norm(s)^2) - 0.5 * norm(Fk)^2)
        # Since we have already computed Fk + Jk s during subproblem, we can reuse it if possible
        Fk_pred = Fk + Jk * s
        ΔTk = (0.5 * (dot(Fk, Fk) - dot(Fk_pred, Fk_pred))) - (0.5 * σk * dot(s, s))

        if non_mono_size > 1
            k = mod(stats.iter, non_mono_size) + 1
            obj_vec[k] = stats.objective
            fck_max = maximum(obj_vec)
            numerator = fck_max - fck
        else
            numerator = stats.objective - fck
        end
        ρk = numerator / ΔTk

        # Update σk
        if ρk >= η2
            σk = max(σmin, γ1 * σk)
        elseif ρk < η1
            σk = σk * γ2
        end

        # Acceptance of the new candidate
        if ρk >= η1
            x .= ck
            residual!(nlp, x, Fk)
            set_objective!(stats, fck)
            # Update Jacobian
            jacobian!(nlp, x, Jk)
            # Update gradient
            gk = Jk' * Fk
            norm_gk = norm(gk)
        end

        set_iter!(stats, stats.iter + 1)
        set_time!(stats, time() - start_time)
        set_dual_residual!(stats, norm_gk)
        optimal = norm_gk ≤ ϵ

        if verbose > 0 && mod(stats.iter, verbose) == 0
            @info infoline
            infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_gk σk
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
        cgtol = max(rtol, min(T(0.1), √norm_gk, T(0.9) * cgtol))
        callback(nlp, solver, stats)
        σk = solver.σ
        done = stats.status != :unknown
    end

    set_solution!(stats, x)
    return stats
end