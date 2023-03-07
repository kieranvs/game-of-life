#include <chrono>

#include "solver.hpp"
#include "utils.hpp"

constexpr int dim = 2050;
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
void run(const char* name)
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

    auto alive = count_alive();
    
    printf("%s: %zd ms, alive=%d\n", name, duration_ms, alive);
}

int main()
{
	buf_current = new uint8_t[dim * dim];
	buf_next = new uint8_t[dim * dim];

	run<SolverNaive<dim>>("naive");
}
