using ADNLPModels
using JSOSolvers
using NLPModels, NLPModelsModifiers, SolverCore, SolverTools

f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
nlp = ADNLPModel(f, [-1.2; 1.0])

stats = GenericExecutionStats(nlp)
T = Float64
solver =
  JSOSolvers.GDA(nlp, verbose = 1, η1 = eps(T) / 10, η2 = T(0.99), λ = T(1.5), max_iter = 5000)
# solver = JSOSolvers.iR2(nlp, verbose = 1)
# solver = JSOSolvers.R2(nlp,verbose=1, max_iter=1000)
# stats = SolverCore.solve!(solver, nlp, stats)
# print(stats.status)
