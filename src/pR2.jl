export pR2, pR2Solver

"""
    pR2(nlp; kwargs...)

An inexact stochastic first-order quadratic regularization method for unconstrained optimization.

For advanced usage, first define a `pR2Solver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = pR2Solver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = eps(T)^(1/4)`, `η2 = T(0.95)`: step acceptance parameters.
- `λ = T(2)`, λ > 1 regularization update parameters. 
- `σmin = eps(T)`: step parameter for pR2 algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β = T(0) ∈ [0,1]` is the constant in the momentum term. If `β == 0`, pR2 does not use momentum.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.
- `non_mono_size = 1`: size of the non-monotone behaviour. If `non_mono_size > 1`, the algorithm will use a non-monotone behaviour.

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
stats = pR2(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = pR2Solver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct pR2Solver{T, V} <: AbstractOptimizationSolver
  x::V
  gx::V
  cx::V
  d::V   # used for momentum term
  σ::T
  obj_vec::V # used for non-monotone behaviour
end

function pR2Solver(nlp::AbstractNLPModel{T, V}; non_mono_size = 1) where {T, V}
  x = similar(nlp.meta.x0)
  gx = similar(nlp.meta.x0)
  cx = similar(nlp.meta.x0)
  d = fill!(similar(nlp.meta.x0), 0)
  σ = zero(T) # init it to zero for now 
  obj_vec = fill(typemin(T), non_mono_size)
  return pR2Solver{T, V}(x, gx, cx, d, σ, obj_vec)
end

@doc (@doc pR2Solver) function pR2(
  nlp::AbstractNLPModel{T, V};
  non_mono_size = 1,
  kwargs...,
) where {T, V}
  solver = pR2Solver(nlp, non_mono_size = non_mono_size)
  return solve!(solver, nlp; non_mono_size = non_mono_size, kwargs...)
end

function SolverCore.reset!(solver::pR2Solver{T}) where {T}
  solver.d .= zero(T)
  fill!(solver.obj_vec, typemin(T))
  solver
end
SolverCore.reset!(solver::pR2Solver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::pR2Solver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  η1 = eps(T)^(1 / 4),
  η2 = T(0.99),
  λ = T(2),
  σmin = zero(T), # μmin = σmin to match the paper
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  β::T = T(0),
  verbose::Int = 0,
  non_mono_size = 1,
) where {T, V}
  unconstrained(nlp) || error("pR2 should only be called on unconstrained problems.")

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)
  μmin = σmin

  x = solver.x .= x
  ∇fk = solver.gx
  ck = solver.cx
  d = solver.d
  σk = solver.σ

  set_iter!(stats, 0)
  set_objective!(stats, obj(nlp, x))

  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  μk = 2^round(log2(norm_∇fk + 1)) / norm_∇fk #TODO confirm if this is the correct initialization
  σk = μk * norm_∇fk
  ρk = zero(T)

  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
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
    #TODO user has to select the stopping window such as moving average of size 1 is R2 stopping and moving avegrae of 5 is recommended now 
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      optimal = false, # the user has to set the first order in the callback, we keep it here so if other status happen we put 
      max_eval = max_eval,
      iter = stats.iter,
      max_iter = max_iter,
      max_time = max_time,
    ),
  )

  solver.σ = σk

  done = stats.status != :unknown

  while !done
    if β == 0
      ck .= x .- (∇fk ./ σk)
    else # momentum term
      d .= ∇fk .* (T(1) - β) .+ d .* β
      ck .= x .- (d ./ σk)
    end

    ΔTk = norm_∇fk / μk
    fck = obj(nlp, ck)

    if fck == -Inf
      set_status!(stats, :unbounded)
      break
    end

    if non_mono_size > 1  #non-monotone behaviour
      k = mod(stats.iter, non_mono_size) + 1
      solver.obj_vec[k] = stats.objective
      fck_max = maximum(solver.obj_vec)
      ρk = (fck_max - fck) / (abs(fck_max - fck - ΔTk))
    else
      ρk = (stats.objective - fck) / ΔTk
    end

    # Update regularization parameters and Acceptance of the new candidate
    step_accepted = ρk >= η1 && σk >= η2
    if step_accepted
      μk = max(μmin, μk / λ)
      x .= ck
    else
      μk = μk * λ
    end

    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info infoline
      σ_stat = step_accepted ? "↘" : "↗"
      infoline =
        @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %+7.1e  %1s" stats.iter stats.objective norm_∇fk μk σk ρk σ_stat
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    callback(nlp, solver, stats)

    ∇fk = solver.gx
    norm_∇fk = norm(∇fk)

    set_dual_residual!(stats, norm_∇fk)
    σk = μk * norm_∇fk
    solver.σ = σk

    optimal = norm_∇fk ≤ ϵ

    #Since the user can force the status, we need to check if the user has stopped the algorithm
    if stats.status == :first_order #this is what user set in their callback, in DNN training 
      set_status!(stats, :first_order)
    else
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
    end

    done = stats.status != :unknown
  end
  set_solution!(stats, x)
  return stats
end
