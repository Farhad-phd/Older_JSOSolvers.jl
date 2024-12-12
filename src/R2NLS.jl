using LinearAlgebra
using LinearOperators
export R2NLS, R2NLSSolver
export ShiftedLBFGSSolver

const R2NLS_allowed_subsolvers = [CglsSolver, CrlsSolver, LsqrSolver, LsmrSolver]

"""
    R2NLS(nlp; kwargs...)

TODO add docstring
"""
mutable struct R2NLSSolver{
  T,
  V,
  Op <: AbstractLinearOperator{T},
  Sub <: KrylovSolver{T, T, V},
} <: AbstractOptimizationSolver
x::V
xt::V
temp::V
gx::V
gt::V
Fx::V
rt::V
Av::V
Atv::V
A::Op
subsolver::Sub
obj_vec::V # used for non-monotone behaviour
cgtol::T
end

function R2NLSSolver(
  nlp::AbstractNLPModel{T, V};
  non_mono_size = 1,
  subsolver_type::Type{<:KrylovSolver} = LsmrSolver,
) where {T, V}
  subsolver_type in R2NLS_allowed_subsolvers || error("subproblem solver must be one of $(R2NLS_allowed_subsolvers)")

  nvar = nlp.meta.nvar
  nequ = nlp.nls_meta.nequ

  x = V(undef, nvar)
  xt = V(undef, nvar)
  temp = V(undef, nequ)
  gx = V(undef, nvar)
  gt = V(undef, nvar)
  Fx = V(undef, nequ)
  rt = V(undef, nequ)
  Av = V(undef, nequ)
  Atv = V(undef, nvar)
  A = jac_op_residual!(nlp, x, Av, Atv)
  Op = typeof(A)
  subsolver = subsolver_type(nequ, nvar, V)
  Sub = typeof(subsolver)

  σ = zero(T) # init it to zero for now 
  cgtol = one(T) # must be ≤ 1.0
  obj_vec = fill(typemin(T), non_mono_size)

  return R2NLSSolver{T, V, Op, Sub}(x, xt, temp, gx, gt, Fx, rt, Av, Atv, A, subsolver, obj_vec, cgtol)
end

function SolverCore.reset!(solver::R2NLSSolver{T}) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver
end
function SolverCore.reset!(solver::R2NLSSolver{T}, nlp::AbstractNLPModel) where {T}
  fill!(solver.obj_vec, typemin(T))
  solver.A = jac_op_residual!(nlp, solver.x, solver.Av, solver.Atv)
  solver.cgtol = one(T)
  solver
end

@doc (@doc R2NLSSolver) function R2NLS(
  nlp::AbstractNLPModel{T, V};
  subsolver_type::Type{<:KrylovSolver} = LsmrSolver,
  non_mono_size = 1,
  kwargs...,
) where {T, V}
  solver = R2NLSSolver(nlp; non_mono_size = non_mono_size, subsolver_type = subsolver_type)
  return solve!(solver, nlp; non_mono_size = non_mono_size, kwargs...)
end

function SolverCore.solve!(
  solver::R2NLSSolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  # Fatol::T = zero(T),
  # Frtol::T = zero(T),
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

  n = nlp.nls_meta.nvar
  m = nlp.nls_meta.nequ
  ##############
  
  x = solver.x .= x
  ck = solver.xt
  ∇fk = solver.gx # k-1
  ∇fn = solver.gt #current 
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
    curv = dot(s, Hs)
    ΔTk = (slope + curv) / 2  # since ∇fk is negative, otherwise we had -dot(s, ∇fk)
    # ΔTk = (dot(s, ∇fk) + σk * dot(s, s)) / 2  # since ∇fk is negative, otherwise we had -dot(s, ∇fk)

    ck .= x .+ s
    # ck .+= s
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
    cgtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * cgtol)) 
    callback(nlp, solver, stats) # cgtol needs to be updated in callback
    σk = solver.σ
    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  return stats
end

function subsolve!(R2NLS::R2NLSSolver, s, H, ∇f, atol, cgtol, n, σ, subsolver_verbose)  
  if R2NLS.subsolver_type isa MinresSolver
    minres!(
      R2NLS.subsolver_type,
      H, #A
      ∇f, #b 
      λ = σ,
      itmax = 2*n,
      verbose = subsolver_verbose,
    )
    s .= R2NLS.subsolver_type.x
    # stas = R2NLS.subsolver_type.stats
  elseif R2NLS.subsolver_type isa KrylovSolver
    Krylov.solve!(
      R2NLS.subsolver_type,
      (H + σ * I(n)),
      ∇f,
      atol = atol,
      rtol = cgtol,
      itmax = 2*n,
      verbose = subsolver_verbose,
    )
    s .= R2NLS.subsolver_type.x
    # stas = R2NLS.subsolver_type.stats

  elseif R2NLS.subsolver_type isa ShiftedLBFGSSolver
    solve_shifted_system!(s, H, ∇f, σ)
  else
    error("Unsupported subsolver type")
  end
end


# function subsolve!(R2NLS::R2NLSSolver, s, H, ∇f, atol, cgtol, n, σ, subsolver_verbose)
#   if R2NLS.subsolver_type isa KrylovSolver
#     # Define a shifted operator that applies H + σ * I(n) without allocating
#     shifted_op = LinearOperator(size(H),
#         (v_in, v_out) -> begin
#             mul!(v_out, H, v_in)       # v_out = H * v_in
#             v_out .+= σ * v_in         # v_out += σ * v_in
#         end
#     )
#     Krylov.solve!(
#         R2NLS.subsolver_type,
#         shifted_op,
#         ∇f,
#         atol = atol,
#         rtol = cgtol,
#         itmax = 2*n,
#         verbose = subsolver_verbose,
#     )
#     s .= R2NLS.subsolver_type.x
#   elseif R2NLS.subsolver_type isa MinresSolver
#     # Use the shift parameter λ = σ to avoid allocation in minres!
#     minres!(
#         R2NLS.subsolver_type,
#         H, # A
#         ∇f, # b
#         λ = σ,
#         itmax = 2*n,
#         verbose = subsolver_verbose,
#     )
#     s .= R2NLS.subsolver_type.x
#   elseif R2NLS.subsolver_type isa ShiftedLBFGSSolver
#     solve_shifted_system!(s, H, ∇f, σ)
#   else
#     error("Unsupported subsolver type")
#   end
# end