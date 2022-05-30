# John Conway's Game of Life
I find this simulation to be fascinating; when I first heard about it, I immediately wanted to go implement it in CUDA, so here it is. More info here: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life - also heavily referenced in this video: https://www.youtube.com/watch?v=HeQX2HjkcNo&t=221s

The configurations in serial and parallel run nicely both single-threaded on a CPU and in parallel on a GPU. It outputs frame processing time at the bottom screen; you'll see how much better the graphics card performs.

This was a great exercise of the use of shared memory for optimal performance when reading from a data structure.

## Implementations
### Serial
A serial implementation that runs on a single thread in the CPU. I implemented this first to just get the game working.

To compile: `g++ serial.cpp`. To run: `./.a.out`

### Parallel
Same functionality on a (NVIDIA) graphics card and better performance. Depending on your card's specs, you might need to adjust block size to meet the amount of threads per block it supports. This was developed using a GTX 1060 with 6GB of memory. It performs (with printing commented out) well up to a (square) grid width of 45000 before running out of memory. For actual visual representation, 50-72 block width is sufficient.

To compile: `nvcc parallel.cu`. To run: `./.a.out`
