using LinearAlgebra
using LinearOperators
export R2NLS, R2NLSSolver
export ShiftedLBFGSSolver

const R2NLS_allowed_subsolvers = [CglsSolver, CrlsSolver, LsqrSolver, LsmrSolver]

"""
    R2NLS(nlp; kwargs...)

TODO add docstring
"""
mutable struct R2NLSSolver{T, V, Op <: AbstractLinearOperator{T}, Sub <: KrylovSolver{T, T, V}} <:
               AbstractOptimizationSolver
  x::V
  xt::V
  temp::V
  gx::V
  Fx::V
  rt::V
  Av::V
  Atv::V
  A::Op
  subsolver::Sub
  obj_vec::V # used for non-monotone behaviour
  cgtol::T
  σ::T
  μ::T
end

function R2NLSSolver(
  nlp::AbstractNLSModel{T, V};
  non_mono_size = 1,
  subsolver_type::Type{<:KrylovSolver} = LsmrSolver,
) where {T, V}
  subsolver_type in R2NLS_allowed_subsolvers ||
    error("subproblem solver must be one of $(R2NLS_allowed_subsolvers)")

  nvar = nlp.meta.nvar
  nequ = nlp.nls_meta.nequ

  x = V(undef, nvar)
  xt = V(undef, nvar)
  temp = V(undef, nequ)
  gx = V(undef, nvar)
  Fx = V(undef, nequ)
  rt = V(undef, nequ)
  Av = V(undef, nequ)
  Atv = V(undef, nvar)
  A = jac_op_residual!(nlp, x, Av, Atv)
  Op = typeof(A)
  subsolver = subsolver_type(nequ, nvar, V)
  Sub = typeof(subsolver)

  σ = zero(T)
  μ = zero(T)
  cgtol = one(T) # must be ≤ 1.0
  obj_vec = fill(typemin(T), non_mono_size)

  return R2NLSSolver{T, V, Op, Sub}(
    x,
    xt,
    temp,
    gx,
    Fx,
    rt,
    Av,
    Atv,
    A,
    subsolver,
    obj_vec,
    cgtol,
    σ,
    μ,
  )
end

function SolverCore.reset!(solver::R2NLSSolver{T}) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver
end
function SolverCore.reset!(solver::R2NLSSolver{T}, nlp::AbstractNLSModel) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver.A = jac_op_residual!(nlp, solver.x, solver.Av, solver.Atv)
  solver.cgtol = one(T)
  solver
end

@doc (@doc R2NLSSolver) function R2NLS(
  nlp::AbstractNLSModel{T, V};
  subsolver_type::Type{<:KrylovSolver} = LsmrSolver,
  non_mono_size = 1,
  kwargs...,
) where {T, V}
  solver = R2NLSSolver(nlp; non_mono_size = non_mono_size, subsolver_type = subsolver_type)
  return solve!(solver, nlp; non_mono_size = non_mono_size, kwargs...)
end

function SolverCore.solve!(
  solver::R2NLSSolver{T, V},
  nlp::AbstractNLSModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  Fatol::T = zero(T),
  Frtol::T = zero(T),
  η1 = eps(T)^(1 / 4),
  η2 = T(0.95),
  λ = T(2),
  σmin = zero(T),
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  verbose::Int = 0,
  subsolver_verbose::Int = 0,
  non_mono_size = 1,
) where {T, V}
  unconstrained(nlp) || error("R2NLS should only be called on unconstrained problems.")
  if non_mono_size < 1
    error("non_mono_size must be greater than or equal to 1")
  end

  if !(nlp.meta.minimize)
    error("R2NLS only works for minimization problem")
  end

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)
  μmin = σmin

  n = nlp.nls_meta.nvar
  m = nlp.nls_meta.nequ

  x = solver.x .= x
  xt = solver.xt
  ∇f = solver.gx # k-1
  subsolver = solver.subsolver
  r, rt = solver.Fx, solver.rt
  # s = solver.s #TODO I do not need this 
  cgtol = solver.cgtol
  σk = solver.σ
  μk = solver.μ

  residual!(nlp, x, r)
  f, ∇f = objgrad!(nlp, x, ∇f, r, recompute = false)

  # preallocate storage for products with A and A'
  A = solver.A # jac_op_residual!(nlp, x, Av, Atv)
  mul!(∇f, A', r)

  norm_∇fk = norm(∇f)
  μk = 2^round(log2(norm_∇fk + 1))
  σk = μk * norm_∇fk

  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  ϵF = Fatol + Frtol * 2 * √f

  # Preallocate xt.
  xt = solver.xt
  temp = solver.temp

  optimal = norm_∇fk ≤ ϵ
  small_residual = 2 * √f ≤ ϵF

  set_iter!(stats, 0)
  set_objective!(stats, f)
  set_dual_residual!(stats, norm_∇fk)

  if optimal
    @info("Optimal point found at initial point")
    @info @sprintf "%5s  %9s  %7s  %7s  %7s  %7s  %1s" "iter" "f" "‖∇f‖" "μ" "σ" "ρ" ""
    @info @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %+7.1e  %1s" stats.iter stats.objective norm_∇fk μk σk ρk ""
  end
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info @sprintf "%5s  %9s  %7s  %7s  %7s  %7s  %1s" "iter" "f" "‖∇f‖" "μ" "σ" "ρ" ""
    infoline =
      @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %+7.1e  %1s" stats.iter stats.objective norm_∇fk μk σk ρk ""
  end

  set_status!(
    stats,
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      optimal = optimal,
      max_eval = max_eval,
      iter = stats.iter,
      small_residual = small_residual,
      max_iter = max_iter,
      max_time = max_time,
    ),
  )

  solver.σ = σk
  solver.μ = μk
  solver.cgtol = cgtol

  callback(nlp, solver, stats)

  cgtol = solver.cgtol
  σk = solver.σ
  μk = solver.μ

  done = stats.status != :unknown
  cgtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * cgtol))

  while !done
    temp .= .-r
    Krylov.solve!(
      subsolver,
      A,
      temp,
      atol = atol,
      rtol = cgtol,
      λ = √(σk), # sqrt(σk / 2),  λ ≥ 0 is a regularization parameter.
      itmax = max(2 * (n + m), 50),
      timemax = max_time - stats.elapsed_time,
      verbose = subsolver_verbose,
    )
    s, cg_stats = subsolver.x, subsolver.stats
    norm_s = norm(s)

    # Compute actual vs. predicted reduction.
    # copyaxpy!(n, one(T), s, x, xt) # xt = x + s
    xt .= x .+ s
    mul!(temp, A, s) # do we update A?
    slope = dot(r, temp)
    curv = dot(temp, temp)
    residual!(nlp, xt, rt)
    fck = obj(nlp, x, rt, recompute = false)
    # fck = obj(nlp, xt)

    # ΔTk = (slope + curv ) / 2  # TODO in Youssef paper they use σ/2 * norm(s)^2  - σk^2 * norm_s^2
    ΔTk = -slope - curv / 2  # TODO in Youssef paper they use σ/2 * norm(s)^2  - σk^2 * norm_s^2

    if fck == -Inf
      set_status!(stats, :unbounded)
      break
    end

    if non_mono_size > 1  #non-monotone behaviour
      k = mod(stats.iter, non_mono_size) + 1
      solver.obj_vec[k] = stats.objective
      fck_max = maximum(solver.obj_vec)
      ρk = (fck_max - fck) / (fck_max - fck + ΔTk)
    else
      ρk = (stats.objective - fck) / ΔTk
    end

    # Update regularization parameters and Acceptance of the new candidate
    step_accepted = ρk >= η1 && σk >= η2
    if step_accepted
      μk = max(μmin, μk / λ)
      # update A implicitly
      x .= xt
      r .= rt
      f = fck
      grad!(nlp, x, ∇f)
      set_objective!(stats, fck)
      norm_∇fk = norm(∇f)
    else
      μk = μk * λ
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    cgtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * cgtol))

    solver.σ = σk
    solver.μ = μk
    solver.cgtol = cgtol
    callback(nlp, solver, stats)
    σk = solver.σ
    μk = solver.μ
    cgtol = solver.cgtol

    ∇fk = solver.gx
    norm_∇fk = norm(∇fk)
    set_dual_residual!(stats, norm_∇fk)
    σk = μk * norm_∇fk

    optimal = norm_∇fk ≤ ϵ
    small_residual = 2 * √f ≤ ϵF

    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info infoline
      σ_stat = step_accepted ? "↘" : "↗"
      infoline =
        @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %+7.1e  %1s" stats.iter stats.objective norm_∇fk μk σk ρk σ_stat #TODO print B norm
    end

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        optimal = optimal,
        small_residual = small_residual,
        max_eval = max_eval,
        iter = stats.iter,
        max_iter = max_iter,
        max_time = max_time,
      ),
    )

    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  return stats
end
