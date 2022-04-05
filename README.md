# John Conway's Game of Life
I find this simulation to be fascinating; when I first heard about it, I immediately wanted to go implement it in CUDA, so here it is. More info here: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life

## Implementations
### Serial
A serial implementation that runs on a single thread in the CPU. I implemented this first to just get the game working.

### Parallel
After getting the serial version working, I started working on threading it out on the graphics card. It works! Though as of now (4/4/22) it probably could be cleaned up, and I need to modify it to utilize shared memory when possible for better performance.