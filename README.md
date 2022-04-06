# John Conway's Game of Life
I find this simulation to be fascinating; when I first heard about it, I immediately wanted to go implement it in CUDA, so here it is. More info here: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life

## Implementations
### Serial
A serial implementation that runs on a single thread in the CPU. I implemented this first to just get the game working.

To compile: `g++ serial.cpp`. To run: `./.a.out`

### Parallel
Same functionality on a graphics card and better performance.

To compile: `nvcc parallel.cu`. To run: `./.a.out`
