using SolverTest

# function tests()
#   @testset "Testing NLP solvers" begin
#     @testset "Unconstrained solvers" begin
#       @testset "$name" for (name, solver) in [
#         ("trunk+cg", (nlp; kwargs...) -> trunk(nlp, subsolver_type = CgSolver; kwargs...)),
#         ("lbfgs", lbfgs),
#         ("tron", tron),
#         ("R2", R2),
#         ("pR2", pR2),
#         ("pR2N", pR2N),
#         ("R2N", R2N),
#       ]
#         unconstrained_nlp(solver)
#         multiprecision_nlp(solver, :unc)
#       end
#     end
#     @testset "Bound-constrained solvers" begin
#       @testset "$solver" for solver in [tron]
#         bound_constrained_nlp(solver)
#         multiprecision_nlp(solver, :unc)
#         multiprecision_nlp(solver, :bnd)
#       end
#     end
#   end
#   @testset "Testing NLS solvers" begin
#     @testset "Unconstrained solvers" begin
#       @testset "$name" for (name, solver) in [
#         ("trunk+cgls", (nls; kwargs...) -> trunk(nls, subsolver_type = CglsSolver; kwargs...)), # trunk with cgls due to multiprecision
#         ("trunk full Hessian", (nls; kwargs...) -> trunk(nls, variant = :Newton; kwargs...)),
#         ("tron+cgls", (nls; kwargs...) -> tron(nls, subsolver_type = CglsSolver; kwargs...)),
#         ("tron full Hessian", (nls; kwargs...) -> tron(nls, variant = :Newton; kwargs...)),
#       ]
#         unconstrained_nls(solver)
#         multiprecision_nls(solver, :unc)
#       end
#     end
#     @testset "Bound-constrained solvers" begin
#       @testset "$name" for (name, solver) in [
#         ("tron+cgls", (nls; kwargs...) -> tron(nls, subsolver_type = CglsSolver; kwargs...)),
#         ("tron full Hessian", (nls; kwargs...) -> tron(nls, variant = :Newton; kwargs...)),
#       ]
#         bound_constrained_nls(solver)
#         multiprecision_nls(solver, :unc)
#         multiprecision_nls(solver, :bnd)
#       end
#     end
#   end
# end

# tests()

# include("solvers/trunkls.jl")
# include("incompatible.jl")

function simple_Run()
  T = Float64
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - 3)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])


  statsR2 = R2(nlp)
  println(statsR2.status, statsR2.iter)
  stats = R2N(nlp, subsolver_type = JSOSolvers.ShiftedLBFGSSolver) 
  println(stats.status, stats.iter)

  # @test stats.status == :first_order
  stats2 = R2N(nlp, subsolver_type = CgSolver)
  println(stats2.status, stats2.iter)

  # @test stats2.status == :first_order
  # @test stats2.solution_reliable

  nlp3 = ADNLSModel(x -> [[10 * (x[i + 1] - x[i]^2) for i = 1:(30 - 1)]; [x[i] - 1 for i = 1:(30 - 1)]], collect(1:30) ./ (30 + 1), 2*30 - 2)
  stats3 = R2N(nlp3, subsolver_type = CgSolver)
  println(stats3.status)


  # @test stats3.status == :first_order

end
simple_Run()