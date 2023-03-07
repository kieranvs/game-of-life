#include "utils.hpp"

void place_lidka(int dim, int x, int y, uint8_t* buf_current)
{
	buf_current[x + 1 + dim * (y + 0)] = 1;
	buf_current[x + 0 + dim * (y + 1)] = 1;
	buf_current[x + 2 + dim * (y + 1)] = 1;
	buf_current[x + 1 + dim * (y + 2)] = 1;
	buf_current[x + 8 + dim * (y + 10)] = 1;
	buf_current[x + 8 + dim * (y + 11)] = 1;
	buf_current[x + 8 + dim * (y + 12)] = 1;
	buf_current[x + 6 + dim * (y + 11)] = 1;
	buf_current[x + 5 + dim * (y + 12)] = 1;
	buf_current[x + 6 + dim * (y + 12)] = 1;
	buf_current[x + 4 + dim * (y + 14)] = 1;
	buf_current[x + 5 + dim * (y + 14)] = 1;
	buf_current[x + 6 + dim * (y + 14)] = 1;
}

void place_glider(int dim, int x, int y, uint8_t* buf_current)
{
	buf_current[x + 0 + dim * (y + 0)] = 1;
    buf_current[x + 1 + dim * (y + 0)] = 1;
    buf_current[x + 2 + dim * (y + 0)] = 1;
    buf_current[x + 2 + dim * (y + 1)] = 1;
    buf_current[x + 1 + dim * (y + 2)] = 1;
}