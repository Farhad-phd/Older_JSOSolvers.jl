export iR2, iR2Solver

"""
    iR2(nlp; kwargs...)

An inexact stochastic first-order quadratic regularization method for unconstrained optimization.

For advanced usage, first define a `iR2Solver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = iR2Solver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = eps(T)^(1/4)`, `η2 = T(0.95)`: step acceptance parameters.
- `λ = T(2)`, λ > 1 regularization update parameters. 
- `σmin = eps(T)`: step parameter for iR2 algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β = T(0) ∈ [0,1]` is the constant in the momentum term. If `β == 0`, iR2 does not use momentum.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.

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
stats = iR2(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = iR2Solver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct iR2Solver{T, V} <: AbstractOptimizationSolver
  x::V
  gx::V
  cx::V
  d::V   # used for momentum term
  σ::T
end

function iR2Solver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  gx = similar(nlp.meta.x0)
  cx = similar(nlp.meta.x0)
  d = fill!(similar(nlp.meta.x0), 0)
  σ= zero(T) # init it to zero for now 
  return iR2Solver{T, V}(x, gx, cx, d, σ)
end

@doc (@doc iR2Solver) function iR2(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = iR2Solver(nlp)
  return solve!(solver, nlp; kwargs...)
end

function SolverCore.reset!(solver::iR2Solver{T}) where {T}
  solver.d .= zero(T)
  solver
end
SolverCore.reset!(solver::iR2Solver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::iR2Solver{T, V},
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
) where {T, V}
  unconstrained(nlp) || error("iR2 should only be called on unconstrained problems.")
  
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
  σk =  μk * norm_∇fk
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
    infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %+7.1e  %1s" stats.iter stats.objective norm_∇fk μk σk ρk ""
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

  while !done
    set_objective!(stats, obj(nlp, x))
    grad!(nlp, x, ∇fk)
    norm_∇fk = norm(∇fk)
    σk =  μk * norm_∇fk 
    
    if β == 0
      ck .= x .- (∇fk ./ σk)
    else # momentum term
      d .= ∇fk .* (T(1) - β) .+ d .* β
      ck .= x .- (d ./ σk)
    end

    ΔTk = norm_∇fk * μk
    fck = obj(nlp, ck)

    if fck == -Inf
      set_status!(stats, :unbounded)
      break
    end

    ρk = (stats.objective - fck) / ΔTk
    # Update regularization parameters and Acceptance of the new candidate
    if ρk >= η1 &&  norm_∇fk >= η2/μk # TODO if we move the μ^-1 to the left side 
      μk = max(μmin,  μk / λ )
      x .= ck
      set_objective!(stats, fck)
      grad!(nlp, x, ∇fk)
      norm_∇fk = norm(∇fk)
      set_dual_residual!(stats, norm_∇fk)
    else
      μk = μk * λ
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    optimal = norm_∇fk ≤ ϵ
    
    σk = μk * norm_∇fk 
    solver.σ = σk

    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info infoline
      σ_stat = (ρk >= η1 &&  norm_∇fk >= η2/μk) ? "↘" : "↗" 
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e  %7.1e  %+7.1e  %1s" stats.iter stats.objective norm_∇fk μk σk ρk σ_stat
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
    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  return stats
end