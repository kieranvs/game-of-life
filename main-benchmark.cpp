#include <chrono>
#include <optional>
#include <algorithm>

#include "solver.hpp"
#include "reference.hpp"
#include "utils.hpp"

constexpr int dim = 4098;
uint8_t* buf_current;
uint8_t* buf_next;

uint32_t count_alive()
{
	uint32_t alive = 0;
	for (int i = 0; i < dim * dim; i++)
		alive += buf_current[i];

	return alive;
}

template <typename Solver>
int64_t run(const char* name, std::optional<int64_t> ref_timing = std::nullopt)
{
	int64_t best_time = std::numeric_limits<int64_t>::max();
	int num_runs = 0;

	while (num_runs < 3 || (num_runs < 10 && best_time < 500))
	{
		for (int i = 0; i < dim * dim; i++)
			buf_current[i] = 0;

	    for (int i = 0; i < dim * dim; i++)
			buf_next[i] = 0;

	    place_lidka(dim, 1000, 1000, buf_current);

	    auto start_timepoint = std::chrono::high_resolution_clock::now();

	    Solver solver;

	    for (int gen = 0; gen < 100; gen++)
	    {
			solver.update(buf_current, buf_next);
			std::swap(buf_current, buf_next);
	    }

	    auto end_timepoint = std::chrono::high_resolution_clock::now();

		auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_timepoint - start_timepoint).count();

		best_time = std::min(best_time, duration_ms);
		num_runs += 1;
	}

    auto alive = count_alive();
    
    if (ref_timing.has_value())
    {
		float speedup = (float)ref_timing.value() / (float)best_time;
		printf("%s: %zd ms, alive=%d, runs=%d, speedup=%.1fx\n", name, best_time, alive, num_runs, speedup);
    }
    else
		printf("%s: %zd ms, alive=%d, runs=%d\n", name, best_time, alive, num_runs);

    return best_time;
}

int main()
{
	buf_current = new uint8_t[dim * dim];
	buf_next = new uint8_t[dim * dim];

	auto ref_timing = run<SolverReference<dim>>("reference");
	run<SolverNaive<dim>>("naive", ref_timing);
    run<SolverBorder<dim>>("border", ref_timing);
    run<SolverAVX<dim>>("simd", ref_timing);

    // run<SolverMT_Busy<dim, 2>>("mt-busy-2", ref_timing);
    // run<SolverMT_Busy<dim, 4>>("mt-busy-4", ref_timing);
    // run<SolverMT_Busy<dim, 6>>("mt-busy-6", ref_timing);
    // run<SolverMT_Busy<dim, 8>>("mt-busy-8", ref_timing);
    // run<SolverMT_Busy<dim, 10>>("mt-busy-10", ref_timing);
    // run<SolverMT_Busy<dim, 12>>("mt-busy-12", ref_timing);
    // run<SolverMT_Busy<dim, 14>>("mt-busy-14", ref_timing);
    // run<SolverMT_Busy<dim, 16>>("mt-busy-16", ref_timing);
    run<SolverMT_Busy<dim, 18>>("mt-busy-18", ref_timing);
    run<SolverMT_Busy<dim, 20>>("mt-busy-20", ref_timing);

    // run<SolverMT<dim, 2, false>>("mt-2", ref_timing);
    // run<SolverMT<dim, 4, false>>("mt-4", ref_timing);
    // run<SolverMT<dim, 6, false>>("mt-6", ref_timing);
    // run<SolverMT<dim, 8, false>>("mt-8", ref_timing);
    // run<SolverMT<dim, 10, false>>("mt-10", ref_timing);
    // run<SolverMT<dim, 12, false>>("mt-12", ref_timing);
    // run<SolverMT<dim, 14, false>>("mt-14", ref_timing);
    // run<SolverMT<dim, 16, false>>("mt-16", ref_timing);
    // run<SolverMT<dim, 18, false>>("mt-18", ref_timing);
    run<SolverMT<dim, 20, false>>("mt-20", ref_timing);

    run<SolverMT<dim, 2, true>>("mt-simd-2", ref_timing);
    run<SolverMT<dim, 4, true>>("mt-simd-4", ref_timing);
    run<SolverMT<dim, 6, true>>("mt-simd-6", ref_timing);
    run<SolverMT<dim, 8, true>>("mt-simd-8", ref_timing);
    run<SolverMT<dim, 10, true>>("mt-simd-10", ref_timing);
    run<SolverMT<dim, 12, true>>("mt-simd-12", ref_timing);
    run<SolverMT<dim, 14, true>>("mt-simd-14", ref_timing);
    run<SolverMT<dim, 16, true>>("mt-simd-16", ref_timing);
    run<SolverMT<dim, 18, true>>("mt-simd-18", ref_timing);
    run<SolverMT<dim, 20, true>>("mt-simd-20", ref_timing);

    return 0;
}
