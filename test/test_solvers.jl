using SolverTest
using BenchmarkTools
using Profile
using ProfileView

# function tests()
#   @testset "Testing NLP solvers" begin
#     @testset "Unconstrained solvers" begin
#       @testset "$name" for (name, solver) in [
#         ("trunk+cg", (nlp; kwargs...) -> trunk(nlp, subsolver_type = CgSolver; kwargs...)),
#         ("lbfgs", lbfgs),
#         ("tron", tron),
#         ("R2", R2),
#         ("fomo_r2", fomo),
#         ("fomo_tr", (nlp; kwargs...) -> fomo(nlp, step_backend = JSOSolvers.tr_step(); kwargs...)),
#       ]
#         unconstrained_nlp(solver)
#         multiprecision_nlp(solver, :unc)
#       end
#       @testset "$name : nonmonotone configuration" for (name, solver) in [
#         ("R2", (nlp; kwargs...) -> R2(nlp, M = 2; kwargs...)),
#         ("fomo_r2", (nlp; kwargs...) -> fomo(nlp, M = 2; kwargs...)),
#         (
#           "fomo_tr",
#           (nlp; kwargs...) -> fomo(nlp, M = 2, step_backend = JSOSolvers.tr_step(); kwargs...),
#         ),
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
function cb(nlp, solver, stats)
  T = Float64
  rtol = 1e-6
  norm_∇fk = norm(solver.gx)
  solver.cgtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * solver.cgtol))
end

function simple_Run()
  T = Float64
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - 3)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])

  # statsR2 = R2(nlp)
  # println(statsR2.status, statsR2.iter)

  # @test_throws ErrorException R2N(nlp, subsolver_type = JSOSolvers.ShiftedLBFGSSolver) 
  # stats = R2N(LBFGSModel(nlp), subsolver_type = JSOSolvers.ShiftedLBFGSSolver)
  # println(stats.status, stats.iter)

  # stats2 = R2N(nlp, subsolver_type = CgSolver)
  # println(stats2.status, stats2.iter)

  # stats3 = R2N(LBFGSModel(nlp),max_iter = 30,callback = cb)
  # println(stats3.status, stats3.iter)

  # stats4= R2N(nlp)
  # # @test stats.status == :first_order
  # stats2 = R2N(nlp, subsolver_type = CgSolver)
  # println(stats2.status, stats2.iter)

  # @test stats2.status == :first_order
  # @test stats2.solution_reliable

  # nlp3 = ADNLSModel(x -> [[10 * (x[i + 1] - x[i]^2) for i = 1:(30 - 1)]; [x[i] - 1 for i = 1:(30 - 1)]], collect(1:30) ./ (30 + 1), 2*30 - 2)
  # stats3 = R2N(nlp3, subsolver_type = CgSolver)
  # println(stats3.status)

  # Ensure the function allocates no more than expected
  solver = JSOSolvers.ShiftedLBFGSSolver
  println("----------------------")
  println("\tShiftedLBFGSSolver")
  println("----------------------")
  # @benchmark  R2N($nlp, subsolver_type = $solver, max_iter = 1) 
  #warm up
  R2N(LBFGSModel(nlp), subsolver_type = solver)
  b = @benchmark R2N(LBFGSModel($nlp), subsolver_type = $solver, max_iter = 1)
  # b= @benchmark lbfgs($nlp)
  io = IOBuffer()
  show(io, "text/plain", b)
  s = String(take!(io))
  println(s)
  # Analyze results
  println(b)
  b1 = @ballocated R2N(LBFGSModel($nlp), subsolver_type = $solver, max_iter = 1)
  println("b1 = ", b1)



  #Testinb with TrunkSolver
  # reset the nlp
  reset!(nlp)
  # warm up 
  trunk(LBFGSModel(nlp))
  println("----------------------")
  println("\tTrunkSolver")
  println("----------------------")
  b = @benchmark trunk(LBFGSModel($nlp))
  io = IOBuffer()
  show(io, "text/plain", b)
  s = String(take!(io))
  println(s)
  println(b)

  b2 = @ballocated trunk(LBFGSModel($nlp))
  println("b2 = ", b2)


  solver = CgSolver
  # reset the nlp
  reset!(nlp)
  # warm up 
  R2N(LBFGSModel(nlp), subsolver_type = solver)

  println("----------------------")
  println("\tCgSolver")
  println("----------------------")

  b = @benchmark R2N($nlp, subsolver_type = $solver, max_iter = 1)
  io = IOBuffer()
  show(io, "text/plain", b)
  s = String(take!(io))
  println(s)
  println(b)

  b3 = @ballocated R2N($nlp, subsolver_type = $solver, max_iter = 1)
  println("b3 = ", b3)
  # solver = MinresSolver
  # println("MinresSolver")
  # b = @benchmark R2N($nlp, subsolver_type = $solver, max_iter = 1)
  # io = IOBuffer()
  # show(io, "text/plain", b)
  # s = String(take!(io))
  # println(s)
  # println(b)

  # Optional: Uncomment if you want to use ProfileView for visualization
  # using ProfileView

  # # Define the solver and model
  # solver = JSOSolvers.ShiftedLBFGSSolver
  # model = LBFGSModel(nlp)  # Ensure `nlp` is defined

  # # Profile the R2N function
  # Profile.clear()  # Clear previous profiling data
  # @profile R2N(model, subsolver_type = solver)

  # # Analyze and display profiling results
  # println("Profiling Results:")
  # Profile.print()
  # ProfileView.view()

end
simple_Run()

# using Profile, PProf
# Profile.Allocs.clear()
# Profile.Allocs.@profile R2N(LBFGSModel(nlp), subsolver_type = JSOSolvers.ShiftedLBFGSSolver)
# PProf.Allocs.pprof()
