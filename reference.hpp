#include <stdint.h>

template <int dim>
struct SolverReference
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