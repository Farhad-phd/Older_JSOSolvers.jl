using LinearAlgebra
using LinearOperators
export R2N, R2NSolver
export ShiftedLBFGSSolver
#TODO check isa(nlp, LSR1Model) and updated op

# const R2N_allowed_subsolvers = [CglsSolver, CrlsSolver, LsqrSolver, LsmrSolver, minres]

#TODO do I export ShiftedLBFGSSolver? in seprate file
# NOTES: the memeory can be define in LBFGSModels by using mem = 5

abstract type AbstractShiftedLBFGSSolver end

struct ShiftedLBFGSSolver <: AbstractShiftedLBFGSSolver
  # Shifted LBFGS-specific fields
end

"""
    R2N(nlp; kwargs...)

A first-order quadratic regularization method for unconstrained optimization.

For advanced usage, first define a `R2NSolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = R2NSolver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = eps(T)^(1/4)`, `η2 = T(0.95)`: step acceptance parameters.
- `γ1 = T(1/2)`, `γ2 = 1/γ1`: regularization update parameters.
- `σmin = eps(T)`: step parameter for R2N algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β = T(0) ∈ [0,1]` is the constant in the momentum term. If `β == 0`, R2N does not use momentum.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `subsolver_type::Union{Type{<:KrylovSolver}, Type{ShiftedLBFGSSolver}} = ShiftedLBFGSSolver`: the subsolver to solve the shifted system. Default is `JSOSolvers.ShiftedLBFGSSolver` which is the exact solver.
- `subsolver_verbose::Int = 0`: if > 0, display iteration information every `subsolver_verbose` iteration of the subsolver if CG is selected.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.gx`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.dual_feas`: norm of current gradient;
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.

# Examples
```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
stats = R2N(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = R2NSolver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct R2NSolver{
  T,
  V,
  Op <: AbstractLinearOperator{T},
  Sub <: Union{KrylovSolver{T, T, V}, ShiftedLBFGSSolver},
} <: AbstractOptimizationSolver
  x::V
  cx::V
  gx::V
  gn::V
  σ::T
  H::Op
  Hs::V
  s::V
  obj_vec::V # used for non-monotone behaviour
  subsolver_type::Sub
  cgtol::T
end

function R2NSolver(
  nlp::AbstractNLPModel{T, V};
  non_mono_size = 1,
  subsolver_type::Union{Type{<:KrylovSolver}, Type{ShiftedLBFGSSolver}} = MinresSolver,
) where {T, V}
  nvar = nlp.meta.nvar
  x = V(undef, nvar)
  cx = V(undef, nvar)
  gx = V(undef, nvar)
  gn = isa(nlp, QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0)
  Hs = V(undef, nvar)
  H = isa(nlp, QuasiNewtonModel) ? nlp.op : hess_op!(nlp, x, Hs)
  Op = typeof(H)
  σ = zero(T) # init it to zero for now 
  s = V(undef, nvar)
  cgtol = one(T) # must be ≤ 1.0
  obj_vec = fill(typemin(T), non_mono_size)
  subsolver =
    isa(subsolver_type, Type{ShiftedLBFGSSolver}) ? subsolver_type() : subsolver_type(nvar, nvar, V)

  Sub = typeof(subsolver)
  return R2NSolver{T, V, Op, Sub}(x, cx, gx, gn, σ, H, Hs, s, obj_vec, subsolver, cgtol)
end

function SolverCore.reset!(solver::R2NSolver{T}) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver
end
function SolverCore.reset!(solver::R2NSolver{T}, ::AbstractNLPModel) where {T}
  fill!(solver.obj_vec, typemin(T))
  @assert (length(solver.gn) == 0) || isa(nlp, QuasiNewtonModel)
  solver.H = hess_op!(nlp, solver.x, solver.Hs)
  solver.cgtol = one(T)
  solver
end

@doc (@doc R2NSolver) function R2N(
  nlp::AbstractNLPModel{T, V};
  subsolver_type::Union{Type{<:KrylovSolver}, Type{ShiftedLBFGSSolver}} = MinresSolver,
  non_mono_size = 1,
  kwargs...,
) where {T, V}
  solver = R2NSolver(nlp, non_mono_size = non_mono_size, subsolver_type = subsolver_type)
  return solve!(solver, nlp; non_mono_size = non_mono_size, kwargs...)
end

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
  if non_mono_size < 1
    error("non_mono_size must be greater than or equal to 1")
  end
  if (solver.subsolver_type isa ShiftedLBFGSSolver && !isa(nlp, LBFGSModel))
    error("Unsupported subsolver type, ShiftedLBFGSSolver is only can be used by LBFGSModel")
  end
  if isa(nlp, LSR1Model)
    @info "only solver allowed is trunked CG for LSR1Model"
    solver.subsolver_type = CrSolver
  end

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  n = nlp.meta.nvar
  x = solver.x .= x
  ck = solver.cx
  ∇fk = solver.gx # k-1
  ∇fn = solver.gn #current 
  s = solver.s
  H = solver.H
  Hs = solver.Hs
  σk = solver.σ
  cgtol = solver.cgtol

  set_iter!(stats, 0)
  set_objective!(stats, obj(nlp, x))

  grad!(nlp, x, ∇fk)
  isa(nlp, QuasiNewtonModel) && (∇fn .= ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  σk = 2^round(log2(norm_∇fk + 1))

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

  solver.σ = σk #TODO do I meed this 
  callback(nlp, solver, stats)
  σk = solver.σ

  done = stats.status != :unknown
  cgtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * cgtol))

  while !done
    ∇fk .*= -1
    subsolve!(solver, s, H, ∇fk, 0.0, cgtol, n, σk, subsolver_verbose)
    slope = dot(n, s, ∇fk) # = -dot(s, ∇fk) but ∇fk is negative
    mul!(Hs, H, s)
    curv = dot(n, s, Hs)
    ΔTk = (slope + curv) / 2  # since ∇fk is negative, otherwise we had -dot(s, ∇fk)
    # ΔTk = (dot(s, ∇fk) + σk * dot(s, s)) / 2  # since ∇fk is negative, otherwise we had -dot(s, ∇fk)

    ck .= x
    ck .+= s
    fck = obj(nlp, ck)
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

    # Update regularization parameters
    if ρk >= η2
      σk = max(σmin, γ1 * σk)
    elseif ρk < η1
      σk = σk * γ2
    end

    # Acceptance of the new candidate
    if ρk >= η1
      x .= ck
      grad!(nlp, x, ∇fk)
      if isa(nlp, QuasiNewtonModel)
        ∇fn .-= ∇fk
        ∇fn .*= -1  # = ∇f(xₖ₊₁) - ∇f(xₖ)
        push!(nlp, s, ∇fn)
        ∇fn .= ∇fk
      end
      set_objective!(stats, fck)
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
    callback(nlp, solver, stats) # cgtol needs to be updated in callback
    # cgtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * cgtol)) 
    σk = solver.σ
    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  return stats
end

function subsolve!(R2N::R2NSolver, s, H, ∇f, atol, cgtol, n, σ, subsolver_verbose)
  if R2N.subsolver_type isa CgSolver
    Krylov.solve!(
      R2N.subsolver_type,
      (H + σ * I(n)),
      ∇f,
      atol = atol,
      rtol = cgtol,
      itmax = max(2 * n, 50),
      verbose = subsolver_verbose,
    )
    s .= R2N.subsolver_type.x
    # stas = R2N.subsolver_type.stats
  elseif R2N.subsolver_type isa MinresSolver
    minres!(
      R2N.subsolver_type,
      H, #A
      ∇f, #b 
      λ = σ,
      # itmax = max(2 * n, 50),
      verbose = subsolver_verbose,
    )
    s .= R2N.subsolver_type.x
    # stas = R2N.subsolver_type.stats
  elseif R2N.subsolver_type isa ShiftedLBFGSSolver
    solve_shifted_system!(s, H, ∇f, σ)
  else
    error("Unsupported subsolver type")
  end
end
