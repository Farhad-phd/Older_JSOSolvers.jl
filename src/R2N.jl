using LinearOperators
export R2N, R2NSolver

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
mutable struct R2NSolver{T, V,Op <: AbstractLinearOperator{T}} <: AbstractOptimizationSolver
  x::V
  gx::V
  cx::V
  σ::T
  B::Op
  s::V
  gt::V
  obj_vec::V # used for non-monotone behaviour
end

function R2NSolver(nlp::AbstractNLPModel{T, V}; mem::Int = 5, non_mono_size = 1) where {T, V}
  nvar = nlp.meta.nvar
  x = similar(nlp.meta.x0)
  gx = similar(nlp.meta.x0)
  cx = similar(nlp.meta.x0)
  σ = zero(T) # init it to zero for now 
  B = LBFGSOperator(T, nvar, mem = mem, scaling = true)
  s = similar(nlp.meta.x0)
  gt = similar(nlp.meta.x0)
  Op = typeof(B)
  obj_vec = fill(typemin(T), non_mono_size)
  return R2NSolver{T, V, Op}(x, gx, cx, σ, B, s, gt,  obj_vec)
end

@doc (@doc R2NSolver) function R2N(nlp::AbstractNLPModel{T, V};   non_mono_size = 1,mem::Int = 5, kwargs...) where {T, V}
  solver = R2NSolver(nlp, mem = mem, non_mono_size = non_mono_size)
  return solve!(solver, nlp;  non_mono_size = non_mono_size, kwargs...) #TODO we don't need to pass  mem::Int = 5, since it will be B.Data.mem
end

function SolverCore.reset!(solver::R2NSolver{T}) where {T}
  solver.d .= zero(T)
  reset!(solver.B)
  fill!(solver.obj_vec, typemin(T))
  solver
end
SolverCore.reset!(solver::R2NSolver, ::AbstractNLPModel) = reset!(solver)

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
  non_mono_size = 1
) where {T, V}
  unconstrained(nlp) || error("R2N should only be called on unconstrained problems.")

  reset!(stats)
  start_time = time()
  set_time!(stats, 0.0)

  x = solver.x .= x
  ∇fk = solver.gx # k-1
  ∇ft = solver.gt #current 
  s = solver.s
  B = solver.B
  ck = solver.cx
  σk = solver.σ
  reset!(B) #TODO do I need this here?

  set_iter!(stats, 0)
  set_objective!(stats, obj(nlp, x))

  grad!(nlp, x, ∇fk)
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

  solver.σ = σk
  callback(nlp, solver, stats)
  σk = solver.σ

  done = stats.status != :unknown

  while !done

    solve_shifted_system!(s, B, -∇fk, σk)
    slope = dot(s , ∇fk)
    curv = dot(s, -(∇fk + σk.* s))
    ΔTk = -slope - curv / 2
    ck .= x .+ s
    fck = obj(nlp, ck)
    if fck == -Inf
      set_status!(stats, :unbounded)
      break
    end

    if non_mono_size > 1  #non-monotone behaviour
      k = mod(stats.iter, non_mono_size) + 1
      solver.obj_vec[k] = stats.objective
      fck_max = maximum(solver.obj_vec)
      ρk = (fck_max - fck) / ( fck_max - fck + ΔTk)
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
      #Update L-BFGS
      grad!(nlp, x, ∇ft)
      @. ∇fk = ∇ft - ∇fk # y = ∇f(xk+1) - ∇f(xk)  # we will update the ∇fk later here
      push!(B, s, ∇fk)
      set_objective!(stats, fck)
      # grad!(nlp, x, ∇fk)
      @. ∇fk = ∇ft # copy ∇ft to ∇fk
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
    callback(nlp, solver, stats) #TODO check if we need to update the ∇fk here
    # Similar to pR2N
    # ∇fk = solver.gx
    # norm_∇fk = norm(∇fk)
    σk = solver.σ

    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  return stats
end
