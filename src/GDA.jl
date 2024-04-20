export GDA, GDASolver

"""
    GDA(nlp; kwargs...)

An Gradient Descent Algorithm (GDA) method for unconstrained optimization.


For advanced usage, first define a `GDASolver` to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = GDASolver(nlp)
    solve!(solver, nlp; kwargs...)

# Arguments
- `nlp::AbstractNLPModel{T, V}` is the model to solve, see `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance: algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `η1 = eps(T)^(1/4)`, `η2 = T(0.95)`: step acceptance parameters.
- `λ = T(2)`, λ > 1 regularization update parameters. 
- `σmin = eps(T)`: step parameter for GDA algorithm.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `β = T(0) ∈ [0,1]` is the constant in the momentum term. If `β == 0`, GDA does not use momentum.
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
stats = GDA(nlp)

# output

"Execution stats: first-order stationary"
```

```jldoctest
using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3))
solver = GDASolver(nlp);
stats = solve!(solver, nlp)

# output

"Execution stats: first-order stationary"
```
"""
mutable struct GDASolver{T, V} <: AbstractOptimizationSolver
  x::V
  gx::V
  cx::V
  d::V   # used for momentum term
  σ::T
end

function GDASolver(nlp::AbstractNLPModel{T, V}) where {T, V}
  x = similar(nlp.meta.x0)
  gx = similar(nlp.meta.x0)
  cx = similar(nlp.meta.x0)
  d = fill!(similar(nlp.meta.x0), 0)
  σ = zero(T) # init it to zero for now 
  return GDASolver{T, V}(x, gx, cx, d, σ)
end

@doc (@doc GDASolver) function GDA(nlp::AbstractNLPModel{T, V}; kwargs...) where {T, V}
  solver = GDASolver(nlp)
  return solve!(solver, nlp; kwargs...)
end

function SolverCore.reset!(solver::GDASolver{T}) where {T}
  solver.d .= zero(T)
  solver
end
SolverCore.reset!(solver::GDASolver, ::AbstractNLPModel) = reset!(solver)

function SolverCore.solve!(
  solver::GDASolver{T, V},
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
  unconstrained(nlp) || error("GDA should only be called on unconstrained problems.")

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)
  μmin = σmin

  x = solver.x .= x
  ∇fk = solver.gx


  set_iter!(stats, 0)
  set_objective!(stats, obj(nlp, x))

  grad!(nlp, x, ∇fk)
  norm_∇fk = norm(∇fk)
  set_dual_residual!(stats, norm_∇fk)


  # Stopping criterion: 
  ϵ = atol + rtol * norm_∇fk
  optimal = norm_∇fk ≤ ϵ
  if optimal
    @info("Optimal point found at initial point")
    @info @sprintf "%5s  %9s  %7s " "iter" "f" "‖∇f‖"
    @info @sprintf "%5d  %9.2e  %7.1e  " stats.iter stats.objective norm_∇fk  
  end
  if verbose > 0 && mod(stats.iter, verbose) == 0
    @info @sprintf "%5s  %9s  %7s " "iter" "f" "‖∇f‖"
    infoline =
      @sprintf "%5d  %9.2e  %7.1e  " stats.iter stats.objective norm_∇fk 
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


  done = stats.status != :unknown

  while !done
    x .= x .- (∇fk ./ σk)
   
    if verbose > 0 && mod(stats.iter, verbose) == 0
      @info infoline
      infoline =
        @sprintf "%5d  %9.2e  %7.1e" stats.iter stats.objective norm_∇fk 
    end

    callback(nlp, solver, stats)
    #since our mini-batch may have changed the values of the gradient, we need to recompute it
    set_objective!(stats, obj(nlp, x))
    grad!(nlp, x, ∇fk)
    norm_∇fk = norm(∇fk)
    set_dual_residual!(stats, norm_∇fk)

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)
    optimal = norm_∇fk ≤ ϵ

    #Since the user can force the status to be something else, we need to check if the user has stopped the algorithm
    if stats.status == :first_order #this is what user set in their callback
      set_status!(stats, :first_order)
    else
      set_status!(
        stats,
        get_status(
          nlp,
          elapsed_time = stats.elapsed_time,
          optimal = false, # the user has to set the first order in the callback
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
