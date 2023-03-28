#include <stdint.h>

__global__ void kernel(uint8_t* buf_current, uint8_t* buf_next, int dim)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int j = blockDim.y * blockIdx.y + threadIdx.y + 1;

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

void run_kernel(uint8_t* buf_current, uint8_t* buf_next, int dim)
{
	int grid_sz = (dim - 2) / 32;
	dim3 block_dim(32, 32, 1);
	dim3 grid_dim(grid_sz, grid_sz, 1);
	kernel<<<grid_dim, block_dim>>>(buf_current, buf_next, dim);
}