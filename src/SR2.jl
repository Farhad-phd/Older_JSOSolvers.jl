export SR2, SR2Solver

"""
    SR2(nlp; kwargs...)

A stochastic first-order quadratic regularization method for unconstrained optimization.

For advanced usage, first define a `SR2Solver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = SR2Solver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = eps(T)^(1/4)`, `η2 = T(0.95)`: step acceptance parameters.
- `γ1 = T(1/2)`, `γ2 = 1/γ1`: regularization update parameters. #TODO   we can use λ but here we only need γ1
- `μmin = eps(T)`: step parameter for SR2 algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β = T(0) ∈ [0,1]` is the constant in the momentum term. If `β == 0`, SR2 does not use momentum.
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
stats = SR2(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = SR2Solver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct SR2Solver{T, V} <: AbstractOptimizationSolver
  x::V
  gx::V
  cx::V
  d::V   # used for momentum term
  μ::T
end

function SR2Solver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  gx = similar(nlp.meta.x0)
  cx = similar(nlp.meta.x0)
  d = fill!(similar(nlp.meta.x0), 0)
  μ= zero(T) # init it to zero for now 
  return SR2Solver{T, V}(x, gx, cx, d, μ)
end

@doc (@doc SR2Solver) function SR2(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = SR2Solver(nlp)
  return solve!(solver, nlp; kwargs...)
end

function SolverCore.reset!(solver::SR2Solver{T}) where {T}
  solver.d .= zero(T)
  solver
end
SolverCore.reset!(solver::SR2Solver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::SR2Solver{T, V},
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
  β::T = T(0),
  verbose::Int = 0,
) where {T, V}
  unconstrained(nlp) || error("SR2 should only be called on unconstrained problems.")
  
  μmin = σmin # we only add this so it matches the R2 

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  x = solver.x .= x
  ∇fk = solver.gx
  ck = solver.cx
  d = solver.d
  μk = solver.μ

  set_iter!(stats, 0)
  set_objective!(stats, obj(nlp, x))

  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)

  μk = 2^round(log2(norm_∇fk + 1))
  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  if optimal
    @info("Optimal point found at initial point")
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "μ"
    @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk μk
  end
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" "μ"
    infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk μk
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

  solver.μ= μk
  callback(nlp, solver, stats)
  μk = solver.μ

  done = stats.status != :unknown

  while !done
    #TODO unlike R2 since our data is stochastic we need to recompute the gradient and objective and not used the passed 
    # set_objective!(stats, obj(nlp, x))
    # grad!(nlp, x, ∇fk)
    # norm_∇fk = norm(∇fk)
    # set_dual_residual!(stats, norm_∇fk)
    # optimal = norm_∇fk ≤ ϵ #todo we need to check
    # we will be slower but more accurate  and no need to do them in the callback 
    
    σk = μk * norm_∇fk # this is different from R2

    if β == 0
      ck .= x .- (∇fk ./ σk)
    else # momentum term
      d .= ∇fk .* (T(1) - β) .+ d .* β
      ck .= x .- (d ./ σk)
    end

    ΔTk =  norm_∇fk /μk  #norm_∇fk^2 / σk #TODO confirm if 1/2 is missing here ?
    fck = obj(nlp, ck)

    if fck == -Inf
      set_status!(stats, :unbounded)
      break
    end

    ρk = (stats.objective - fck) / ΔTk

    # Update regularization parameters and Acceptance of the new candidate
    if ρk >= η1 && σk >= η2  # if we move the μ^-1 to the left side 
      x .= ck
      set_objective!(stats, fck)
      grad!(nlp, x, ∇fk)
      norm_∇fk = norm(∇fk)
      μk = max(μmin,  μk * γ2 )
    else
      μk = μk * γ1 
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, norm_∇fk)
    optimal = norm_∇fk ≤ ϵ

    
    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info infoline
      infoline = @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk μk
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
    solver.μ= μk
    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  return stats
end
