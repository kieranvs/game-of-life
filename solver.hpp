#include <immintrin.h>
#include <optional>
#include <thread>
#include <mutex>
#include <algorithm>
#include <condition_variable>

template <int dim>
struct SolverNaive
{
	void update(uint8_t* buf_current, uint8_t* buf_next)
	{
		for (int i = 0; i < dim; i++)
		{
			bool i_low = i == 0;
			bool i_high = i == (dim - 1);

			for (int j = 0; j < dim; j++)
			{
				bool j_low = j == 0;
				bool j_high = j == (dim - 1);

				uint8_t neighbours = 0;
				if (!i_low &&             buf_current[i - 1 + dim * (j    )]) neighbours += 1;
				if (!i_low && !j_low &&   buf_current[i - 1 + dim * (j - 1)]) neighbours += 1;
				if (!i_low && !j_high &&  buf_current[i - 1 + dim * (j + 1)]) neighbours += 1;

				if (!j_low &&             buf_current[i     + dim * (j - 1)]) neighbours += 1;
				if (!j_high &&            buf_current[i     + dim * (j + 1)]) neighbours += 1;

				if (!i_high &&             buf_current[i + 1 + dim * (j    )]) neighbours += 1;
				if (!i_high && !j_low &&   buf_current[i + 1 + dim * (j - 1)]) neighbours += 1;
				if (!i_high && !j_high &&  buf_current[i + 1 + dim * (j + 1)]) neighbours += 1;

				if (buf_current[i + dim * j])
					buf_next[i + dim * j] = (neighbours == 2 || neighbours == 3);
				else
					buf_next[i + dim * j] = (neighbours == 3);
			}
		}
	}
};

template <int dim>
struct SolverBorder
{
	void update(uint8_t* buf_current, uint8_t* buf_next)
	{
		for (int i = 1; i < dim - 1; i++)
		{
			for (int j = 1; j < dim - 1; j++)
			{
				uint8_t neighbours = 0;
				neighbours += buf_current[i - 1 + dim * (j    )];
				neighbours += buf_current[i - 1 + dim * (j - 1)];
				neighbours += buf_current[i - 1 + dim * (j + 1)];

				neighbours += buf_current[i     + dim * (j - 1)];
				neighbours += buf_current[i     + dim * (j + 1)];

				neighbours += buf_current[i + 1 + dim * (j    )];
				neighbours += buf_current[i + 1 + dim * (j - 1)];
				neighbours += buf_current[i + 1 + dim * (j + 1)];

				if (buf_current[i + dim * j])
					buf_next[i + dim * j] = (neighbours == 2 || neighbours == 3);
				else
					buf_next[i + dim * j] = (neighbours == 3);
			}
		}
	}
};

inline __m256i _mm256_not_si256(__m256i a)
{
	return _mm256_xor_si256(a, _mm256_cmpeq_epi32(a, a));
}

template <int dim>
struct SolverAVX
{
	void update(uint8_t* buf_current, uint8_t* buf_next)
	{
		static_assert((dim - 2) % 32 == 0);

		for (int j = 1; j < dim - 1; j++)
		{
			for (int i = 1; i < dim - 1; i += 32)
			{
				__m256i neighbours = _mm256_setzero_si256();
				neighbours = _mm256_add_epi8(neighbours,
					_mm256_loadu_si256((const __m256i*)&buf_current[i - 1 + dim * (j    )]));
				neighbours = _mm256_add_epi8(neighbours,
					_mm256_loadu_si256((const __m256i*)&buf_current[i - 1 + dim * (j - 1)]));
				neighbours = _mm256_add_epi8(neighbours,
					_mm256_loadu_si256((const __m256i*)&buf_current[i - 1 + dim * (j + 1)]));

				neighbours = _mm256_add_epi8(neighbours,
					_mm256_loadu_si256((const __m256i*)&buf_current[i     + dim * (j - 1)]));
				neighbours = _mm256_add_epi8(neighbours,
					_mm256_loadu_si256((const __m256i*)&buf_current[i     + dim * (j + 1)]));

				neighbours = _mm256_add_epi8(neighbours,
					_mm256_loadu_si256((const __m256i*)&buf_current[i + 1 + dim * (j    )]));
				neighbours = _mm256_add_epi8(neighbours,
					_mm256_loadu_si256((const __m256i*)&buf_current[i + 1 + dim * (j - 1)]));
				neighbours = _mm256_add_epi8(neighbours,
					_mm256_loadu_si256((const __m256i*)&buf_current[i + 1 + dim * (j + 1)]));

				__m256i n2 = _mm256_cmpeq_epi8(neighbours, _mm256_set1_epi8(2));
				__m256i n3 = _mm256_cmpeq_epi8(neighbours, _mm256_set1_epi8(3));
				__m256i n23 = _mm256_or_si256(n2, n3);
				__m256i cr = _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)&buf_current[i + dim * j]),
					_mm256_set1_epi8(1));
				__m256i ncr = _mm256_not_si256(cr);
				__m256i a = _mm256_and_si256(cr, n23);
				__m256i b = _mm256_and_si256(ncr, n3);
				__m256i result = _mm256_and_si256(_mm256_or_si256(a, b), _mm256_set1_epi8(1));
				
				_mm256_storeu_si256((__m256i*)&buf_next[i + dim * j], result);
			}
		}
	}
};

template <int dim, int num_threads>
void do_slice(int j_start, uint8_t* buf_current, uint8_t* buf_next)
{
	for (int j = j_start; j < (j_start + (dim / num_threads)); j++)
	{
		bool j_low = j == 0;
		bool j_high = j == (dim - 1);

		for (int i = 0; i < dim; i++)
		{
			bool i_low = i == 0;
			bool i_high = i == (dim - 1);

			uint8_t neighbours = 0;
			if (!i_low &&             buf_current[i - 1 + dim * (j    )]) neighbours += 1;
			if (!i_low && !j_low &&   buf_current[i - 1 + dim * (j - 1)]) neighbours += 1;
			if (!i_low && !j_high &&  buf_current[i - 1 + dim * (j + 1)]) neighbours += 1;

			if (!j_low &&             buf_current[i     + dim * (j - 1)]) neighbours += 1;
			if (!j_high &&            buf_current[i     + dim * (j + 1)]) neighbours += 1;

			if (!i_high &&             buf_current[i + 1 + dim * (j    )]) neighbours += 1;
			if (!i_high && !j_low &&   buf_current[i + 1 + dim * (j - 1)]) neighbours += 1;
			if (!i_high && !j_high &&  buf_current[i + 1 + dim * (j + 1)]) neighbours += 1;

			if (buf_current[i + dim * j])
				buf_next[i + dim * j] = (neighbours == 2 || neighbours == 3);
			else
				buf_next[i + dim * j] = (neighbours == 3);
		}
	}
}

template <int dim, int num_threads>
void do_slice_simd(int j_start, uint8_t* buf_current, uint8_t* buf_next)
{
	static_assert((dim - 2) % 32 == 0);

	for (int j = j_start; j < (j_start + ((dim - 2) / num_threads)); j++)
	{
		for (int i = 1; i < dim - 1; i += 32)
		{
			__m256i neighbours = _mm256_setzero_si256();
			neighbours = _mm256_add_epi8(neighbours,
				_mm256_loadu_si256((const __m256i*)&buf_current[i - 1 + dim * (j    )]));
			neighbours = _mm256_add_epi8(neighbours,
				_mm256_loadu_si256((const __m256i*)&buf_current[i - 1 + dim * (j - 1)]));
			neighbours = _mm256_add_epi8(neighbours,
				_mm256_loadu_si256((const __m256i*)&buf_current[i - 1 + dim * (j + 1)]));

			neighbours = _mm256_add_epi8(neighbours,
				_mm256_loadu_si256((const __m256i*)&buf_current[i     + dim * (j - 1)]));
			neighbours = _mm256_add_epi8(neighbours,
				_mm256_loadu_si256((const __m256i*)&buf_current[i     + dim * (j + 1)]));

			neighbours = _mm256_add_epi8(neighbours,
				_mm256_loadu_si256((const __m256i*)&buf_current[i + 1 + dim * (j    )]));
			neighbours = _mm256_add_epi8(neighbours,
				_mm256_loadu_si256((const __m256i*)&buf_current[i + 1 + dim * (j - 1)]));
			neighbours = _mm256_add_epi8(neighbours,
				_mm256_loadu_si256((const __m256i*)&buf_current[i + 1 + dim * (j + 1)]));

			__m256i n2 = _mm256_cmpeq_epi8(neighbours, _mm256_set1_epi8(2));
			__m256i n3 = _mm256_cmpeq_epi8(neighbours, _mm256_set1_epi8(3));
			__m256i n23 = _mm256_or_si256(n2, n3);
			__m256i cr = _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)&buf_current[i + dim * j]),
				_mm256_set1_epi8(1));
			__m256i ncr = _mm256_not_si256(cr);
			__m256i a = _mm256_and_si256(cr, n23);
			__m256i b = _mm256_and_si256(ncr, n3);
			__m256i result = _mm256_and_si256(_mm256_or_si256(a, b), _mm256_set1_epi8(1));
			
			_mm256_storeu_si256((__m256i*)&buf_next[i + dim * j], result);
		}
	}
}

template <int dim, int num_threads>
struct SolverMT_Busy
{
	struct MT_Control
	{
		uint8_t* buf_current{};
		uint8_t* buf_next{};

	    std::atomic<bool> go_flag[num_threads];
	    std::atomic<bool> done_flag[num_threads];
	    std::atomic<bool> kill_flag[num_threads];
	};

	static void worker_func(MT_Control* ptr, int id)
	{
		auto& control = *ptr;
		
		auto j_start = (dim / num_threads) * id;

		while (true)
		{
			// Wait for the signal to go
			while (control.go_flag[id].load() == false && control.kill_flag[id].load() == false) {}

			if (control.kill_flag[id].load() == true)
				return;

			control.go_flag[id] = false;

			auto buf_current = control.buf_current;
			auto buf_next = control.buf_next;

			do_slice<dim, num_threads>(j_start, buf_current, buf_next);

			control.done_flag[id] = true;
		}
	}

	std::optional<std::thread> threads[num_threads];
	MT_Control control;

	SolverMT_Busy()
	{
		for (int i = 0; i < num_threads; i++)
		{
			control.go_flag[i] = false;
			control.done_flag[i] = false;
			control.kill_flag[i] = false;
		}

		for (int i = 0; i < num_threads; i++)
		{
			threads[i].emplace(worker_func, &control, i);
		}
	}

	~SolverMT_Busy()
	{
		for (int i = 0; i < num_threads; i++)
		{
			control.kill_flag[i] = true;
			threads[i]->join();
		}
	}

	void update(uint8_t* tcurrent, uint8_t* tnext)
	{
		control.buf_current = tcurrent;
		control.buf_next = tnext;

		for (int i = 0; i < num_threads; i++)
		{
			control.go_flag[i] = true;
		}

		auto all_done = [&]()
		{
			bool d = true;
			for (int i = 0; i < num_threads; i++)
				d &= control.done_flag[i].load();
			return d;
		};

		while (!all_done()) {}

		// They're all done now
		for (int i = 0; i < num_threads; i++)
			control.done_flag[i] = false;
	}
};


template <int dim, int num_threads, bool simd>
struct SolverMT
{
	struct MT_Control
	{
		uint8_t* buf_current{};
		uint8_t* buf_next{};

		std::mutex work_available_signal_mutex;
	    std::condition_variable work_available_signal;

	    std::mutex work_done_signal_mutex;
	    std::condition_variable work_done_signal;

	    std::atomic<bool> go_flag[num_threads];
	    std::atomic<bool> done_flag[num_threads];
	    std::atomic<bool> kill_flag[num_threads];
	};

	static void worker_func(MT_Control* ptr, int id)
	{
		auto& control = *ptr;
		
		auto j_start = (dim / num_threads) * id;
		if constexpr (simd)
			j_start = 1 + ((dim - 2) / num_threads) * id;

		while (true)
		{
			// Wait for the signal to go
			{
				std::unique_lock<std::mutex> lock(control.work_available_signal_mutex);

				if (control.go_flag[id].load() == false && control.kill_flag[id].load() == false)
					control.work_available_signal.wait(lock);
			}

			if (control.kill_flag[id].load() == true)
				return;

			if (control.go_flag[id].load() == false)
				continue;

			control.go_flag[id] = false;

			auto buf_current = control.buf_current;
			auto buf_next = control.buf_next;

			if constexpr (simd)
				do_slice_simd<dim, num_threads>(j_start, buf_current, buf_next);
			else
				do_slice<dim, num_threads>(j_start, buf_current, buf_next);

			// Set the done signal
			{
				std::unique_lock<std::mutex> lock(control.work_done_signal_mutex);

				control.done_flag[id] = true;
				control.work_done_signal.notify_all();
			}
		}
	}

	std::optional<std::thread> threads[num_threads];
	MT_Control control;

	SolverMT()
	{
		for (int i = 0; i < num_threads; i++)
		{
			control.go_flag[i] = false;
			control.done_flag[i] = false;
			control.kill_flag[i] = false;
		}

		for (int i = 0; i < num_threads; i++)
		{
			threads[i].emplace(worker_func, &control, i);
		}
	}

	~SolverMT()
	{
		{
			std::lock_guard lock(control.work_available_signal_mutex);

			for (int i = 0; i < num_threads; i++)
				control.kill_flag[i] = true;

		}
			control.work_available_signal.notify_all();

		for (int i = 0; i < num_threads; i++)
		{
			threads[i]->join();
		}
	}

	void update(uint8_t* tcurrent, uint8_t* tnext)
	{
		control.buf_current = tcurrent;
		control.buf_next = tnext;

		// Set the work available signal
		{
			std::lock_guard lock(control.work_available_signal_mutex);
			
			for (int i = 0; i < num_threads; i++)
				control.go_flag[i] = true;

			control.work_available_signal.notify_all();
		}

		// Wait for the done signal
		while (true)
		{
			std::unique_lock lock(control.work_done_signal_mutex);
			bool done = true;
			for (int i = 0; i < num_threads; i++)
				done &= control.done_flag[i].load();
			
			if (done)
				break;

			control.work_done_signal.wait(lock);
		}

		for (int i = 0; i < num_threads; i++)
			control.done_flag[i] = false;
	}
};