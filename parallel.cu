#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <unistd.h>

#define BLOCK_WIDTH 32
#define GRID_SIZE 50

#define gpuErrchk(ans) { gpuAssert((ans), #ans, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* const func, char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
      std::cerr << cudaGetErrorString(code) << " " << func << std::endl;
        if (abort) {
          exit(code);
        }
    }
}

__device__ int getIdx(int x, int y) {
  if (x < 0 || y < 0 || x >= GRID_SIZE || y >= GRID_SIZE) {
    return -1;
  }
  return y * GRID_SIZE + x;
}

__device__ bool assessNeighborCount(int counter, bool currentVal) {
  if (counter < 2) {
    return false;
  }
  if (currentVal) {
    if (counter <= 3) {
      // Survives
      return true;
    }
    // Dies
    return false;
  }
  
  // Cell is currently dead - will it become alive?
  if (counter == 3) {
    // Yes
    return true;
  }

  // No
  return false;
}

__global__ void computeNextGeneration(const bool* const inputGrid, bool* const outputGrid) {
  int2 pos2d = make_int2(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
  int idx = pos2d.y * GRID_SIZE + pos2d.x;

  if (pos2d.x >= GRID_SIZE || pos2d.y > GRID_SIZE) {
    return;
  }

  // __shared__ bool* s_input_grid

  int counter = 0;
  int cellValue;


  int topLeftIdx = getIdx(pos2d.x - 1, pos2d.y - 1);
  if (topLeftIdx >= 0) {
    counter += inputGrid[topLeftIdx] ? 1 : 0;
  }

  int topIdx = getIdx(pos2d.x, pos2d.y - 1);
  if (topIdx >= 0) {
    counter += inputGrid[topIdx] ? 1 : 0;
  }

  int topRightIdx = getIdx(pos2d.x + 1, pos2d.y - 1);
  if (topRightIdx >= 0) {
    counter += inputGrid[topRightIdx] ? 1 : 0;
  }

  int leftIdx = getIdx(pos2d.x - 1, pos2d.y);
  if (leftIdx >= 0) {
    counter += inputGrid[leftIdx] ? 1 : 0;
  }

  int rightIdx = getIdx(pos2d.x + 1, pos2d.y);
  if (rightIdx >= 0) {
    counter += inputGrid[rightIdx] ? 1 : 0;
  }

  int bottomLeftIdx = getIdx(pos2d.x - 1, pos2d.y + 1);
  if (bottomLeftIdx >= 0) {
    counter += inputGrid[bottomLeftIdx] ? 1 : 0;
  }

  int bottomIdx = getIdx(pos2d.x, pos2d.y + 1);
  if (bottomIdx >= 0) {
    counter += inputGrid[bottomIdx] ? 1 : 0;
  }

  int bottomRightIdx = getIdx(pos2d.x + 1, pos2d.y + 1);
  if (bottomRightIdx >= 0) {
    counter += inputGrid[bottomRightIdx] ? 1 : 0;
  }

  // Assess counter value
  outputGrid[idx] = assessNeighborCount(counter, inputGrid[idx]);
}

__global__ void placeCells(bool* const grid, curandState* state) {
  const int2 twoDimCoords = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                      blockIdx.y * blockDim.y + threadIdx.y);
  const int idx = twoDimCoords.y * GRID_SIZE + twoDimCoords.x;

  if (twoDimCoords.x >= GRID_SIZE || twoDimCoords.y >= GRID_SIZE) {
    return;
  }
  int randomVal = (int)(curand_uniform(&state[idx]) * 100.0f);
  if (randomVal > 75) {
    grid[idx] = 1;
  } else {
    grid[idx] = 0;
  }
}

__global__ void initCurand(curandState* state, unsigned long seed) {
  int2 pos2d = make_int2(threadIdx.x + blockDim.x * blockIdx.x, threadIdx.y + blockDim.y * blockIdx.y);
  int idx = pos2d.y * GRID_SIZE + pos2d.x;

  curand_init(seed, idx, 0, &state[idx]);
}

void printGrid(bool* grid) {
  printf("---------------\n");
  for (int i = 0; i < GRID_SIZE; ++i) {
    for (int j = 0; j < GRID_SIZE; ++j) {
      int idx = i * GRID_SIZE + j;
      std::cout << grid[idx] << " ";
    }
    std::cout << "\n";
  }
}

int main(int argc, char** argv) {
  const dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);
  // const dim3 gridSize(1,1,1);
  const dim3 gridSize(ceil(1.0f*GRID_SIZE / blockSize.x), ceil(1.0f*GRID_SIZE / blockSize.y));
  gpuErrchk(cudaFree(0));

  size_t allocSize = sizeof(bool) * GRID_SIZE * GRID_SIZE;
  bool* h_grid = (bool*)malloc(allocSize);
  memset(h_grid, 0, allocSize);
  bool* d_gridInput;
  bool* d_gridOutput;
  gpuErrchk(cudaMalloc(&d_gridInput, allocSize));
  gpuErrchk(cudaMalloc(&d_gridOutput, allocSize));

  unsigned long seed = time(NULL);

  curandState* state;
  gpuErrchk(cudaMalloc((void **)&state, sizeof(curandState) * GRID_SIZE * GRID_SIZE));

  initCurand<<<gridSize, blockSize>>>(state, seed);

  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  placeCells<<<gridSize, blockSize>>>(d_gridInput, state);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  bool* initialGrid = (bool*)malloc(allocSize);
  gpuErrchk(cudaMemcpy(initialGrid, d_gridInput, allocSize, cudaMemcpyDeviceToHost));
  printGrid(initialGrid);

  while (true) {
    bool* output = (bool*)malloc(allocSize);
    computeNextGeneration<<<gridSize, blockSize>>>(d_gridInput, d_gridOutput);
    cudaDeviceSynchronize();

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMemcpy(output, d_gridOutput, allocSize, cudaMemcpyDeviceToHost));
    printGrid(output);

    gpuErrchk(cudaMemcpy(d_gridInput, d_gridOutput, allocSize, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemset(d_gridOutput, 0, allocSize));

    free(output);
    // sleep(1);
    struct timespec tim, tim2;
    tim.tv_sec = 0;
    tim.tv_nsec = 100000000L;
    nanosleep(&tim, &tim2);
  }

  free(h_grid);
  gpuErrchk(cudaFree(state));
  gpuErrchk(cudaFree(d_gridInput));
  gpuErrchk(cudaFree(d_gridOutput));
}