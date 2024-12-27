# using SolverTest
# using BenchmarkTools
using Profile

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

  stats = R2N(LBFGSModel(nlp), subsolver_type = JSOSolvers.ShiftedLBFGSSolver, verbose = 1, max_iter = 100)
  println(stats.status, stats.iter)

  stats2 = R2N(nlp, subsolver_type = CgSolver, verbose = 1, max_iter = 100)
  # println(stats2.status, stats2.iter)

  # stats3 = R2N(LBFGSModel(nlp),max_iter = 30,callback = cb, subsolver_type =Krylov.MinresSolver)
  # println(stats3.status, stats3.iter)

  # stats4= R2N(nlp)
  # @test stats.status == :first_order
#   # stats2 = R2N(nlp, subsolver_type = CgSolver)
#   # println(stats2.status, stats2.iter)

#   # @test stats2.status == :first_order
#   # @test stats2.solution_reliable

#   # nlp3 = ADNLSModel(x -> [[10 * (x[i + 1] - x[i]^2) for i = 1:(30 - 1)]; [x[i] - 1 for i = 1:(30 - 1)]], collect(1:30) ./ (30 + 1), 2*30 - 2)
#   # stats3 = R2N(nlp3, subsolver_type = CgSolver)
#   # println(stats3.status)

#   # Ensure the function allocates no more than expected
#   solver = JSOSolvers.ShiftedLBFGSSolver
#   println("----------------------")
#   println("\tShiftedLBFGSSolver")
#   println("----------------------")
#   # warm up
#   R2N(LBFGSModel(nlp), subsolver_type = solver)
#   b = @benchmark R2N(LBFGSModel($nlp), subsolver_type = $solver, max_iter = 1)
#   # b= @benchmark lbfgs($nlp)
#   io = IOBuffer()
#   show(io, "text/plain", b)
#   s = String(take!(io))
#   println(s)
#   println(b)


  println("MinresSolver")

end
simple_Run()



# T = Float64
# f(x) = (x[1] - 1)^2 + 4 * (x[2] - 3)^2
# nlp = ADNLPModel(f, [-1.2; 1.0])

# solver = JSOSolvers.ShiftedLBFGSSolver

# # # warm up
# R2N(LBFGSModel(nlp), subsolver_type = solver)
# # #   @benchmark R2N(LBFGSModel($nlp), subsolver_type = $solver, max_iter = 1)

# # Profile.clear_malloc_data()
# # R2N(LBFGSModel(nlp), subsolver_type = solver)




# model = ADNLSModel(x -> [10 * (x[2] - x[1]^2), 1 - x[1]], [-2.0, 1.0], 2)

# for subsolver in JSOSolvers.R2NLS_allowed_subsolvers
#     stats = with_logger(NullLogger()) do
#       R2NLS(model, subsolver_type = subsolver)
#     end
#     @test stats.status == :first_order
#     @test stats.solution_reliable
#     isapprox(stats.solution, ones(2), rtol = 1e-4)
#     @test stats.objective_reliable
#     @test isapprox(stats.objective, 0, atol = 1e-6)
#     @test neval_jac_residual(model) == 0
#     stline = statsline(stats, [:objective, :dual_feas, :elapsed_time, :iter, :status])
#     reset!(model)
#   end