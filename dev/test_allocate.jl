using Printf, LinearAlgebra, Logging, SparseArrays, Test

# additional packages
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools
using NLPModelsTest

# this package
using JSOSolvers
using BenchmarkTools
using Profile
using ProfileView

T = Float64
f(x) = (x[1] - 1)^2 + 4 * (x[2] - 3)^2
nlp = ADNLPModel(f, [-1.2; 1.0])

solver = JSOSolvers.ShiftedLBFGSSolver

# warm up
R2N(LBFGSModel(nlp), subsolver_type = solver)
#   @benchmark R2N(LBFGSModel($nlp), subsolver_type = $solver, max_iter = 1)

Profile.clear_malloc_data()
R2N(LBFGSModel(nlp), subsolver_type = solver, max_iter = 2)