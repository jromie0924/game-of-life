#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#include "utils.h"

#define BLOCK_WIDTH 32
#define GRID_SIZE 72

bool continueNextGeneration = true;

/**
 * @brief Determine whether coordinates are within the bounds of shared memory.
 * 
 * @param x x coordinate
 * @param y y coordinate
 * @return boolean
 */
__device__ bool useShared(int x, int y) {
  if (x < 0 || y < 0 || x >= blockDim.x || y >= blockDim.y) {
    return false;
  }
  return true;
}

__device__ int getIdx_global(int x, int y) {
  if (x < 0 || y < 0 || x >= GRID_SIZE || y >= GRID_SIZE) {
    return -1;
  }
  return y * GRID_SIZE + x;
}

__device__ int getIdx_shared(int x, int y) {
  return y * blockDim.x + x;
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

  // Obtain mapped shared memory location
  int2 s_pos2d = make_int2(threadIdx.x, threadIdx.y);
  int s_idx = s_pos2d.y * blockDim.x + s_pos2d.x;

  __shared__ bool s_inputGrid[BLOCK_WIDTH * BLOCK_WIDTH];
  s_inputGrid[s_idx] = inputGrid[idx];
  __syncthreads();

  int counter = 0;

  // Top left
  int topLeftIdx;
  if (useShared(s_pos2d.x - 1, s_pos2d.y - 1)) {
    topLeftIdx = getIdx_shared(s_pos2d.x - 1, s_pos2d.y - 1);
    counter += s_inputGrid[topLeftIdx] ? 1 : 0;
  } else {
    topLeftIdx = getIdx_global(pos2d.x - 1, pos2d.y - 1);
    if (topLeftIdx >= 0) {
      counter += inputGrid[topLeftIdx] ? 1 : 0;
    }
  }

  // Top
  int topIdx;
  if (useShared(s_pos2d.x, s_pos2d.y - 1)) {
    topIdx = getIdx_shared(s_pos2d.x, s_pos2d.y - 1);
    counter += s_inputGrid[topIdx] ? 1 : 0;
  } else {
    topIdx = getIdx_global(pos2d.x, pos2d.y - 1);
    if (topIdx >= 0) {
      counter += inputGrid[topIdx] ? 1 : 0;
    }
  }

  // Top right
  int topRightIdx;
  if (useShared(s_pos2d.x + 1, s_pos2d.y - 1)) {
    topRightIdx = getIdx_shared(s_pos2d.x + 1, s_pos2d.y - 1);
    counter += s_inputGrid[topRightIdx] ? 1 : 0;
  } else {
    topRightIdx = getIdx_global(pos2d.x + 1, pos2d.y - 1);
    if (topRightIdx >= 0) {
      counter += inputGrid[topRightIdx] ? 1 : 0;
    }
  }

  // Left
  int leftIdx;
  if (useShared(s_pos2d.x - 1, s_pos2d.y)) {
    leftIdx = getIdx_shared(s_pos2d.x - 1, s_pos2d.y);
    counter += s_inputGrid[leftIdx] ? 1 : 0;
  } else {
    leftIdx = getIdx_global(pos2d.x - 1, pos2d.y);
    if (leftIdx >= 0) {
      counter += inputGrid[leftIdx] ? 1 : 0;
    }
  }

  // Right
  int rightIdx;
  if (useShared(s_pos2d.x + 1, s_pos2d.y)) {
    rightIdx = getIdx_shared(s_pos2d.x + 1, s_pos2d.y);
    counter += s_inputGrid[rightIdx] ? 1 : 0;
  } else {
    rightIdx = getIdx_global(pos2d.x + 1, pos2d.y);
    if (rightIdx >= 0) {
      counter += inputGrid[rightIdx] ? 1 : 0;
    }
  }

  // Bottom left
  int bottomLeftIdx;
  if (useShared(s_pos2d.x - 1, s_pos2d.y + 1)) {
    bottomLeftIdx = getIdx_shared(s_pos2d.x - 1, s_pos2d.y + 1);
    counter += s_inputGrid[bottomLeftIdx] ? 1 : 0;
  } else {
    bottomLeftIdx = getIdx_global(pos2d.x - 1, pos2d.y + 1);
    if (bottomLeftIdx >= 0) {
      counter += inputGrid[bottomLeftIdx] ? 1 : 0;
    }
  }
  
  // Bottom
  int bottomIdx;
  if (useShared(s_pos2d.x, s_pos2d.y + 1)) {
    bottomIdx = getIdx_shared(s_pos2d.x, s_pos2d.y + 1);
    counter += s_inputGrid[bottomIdx] ? 1 : 0;
  } else {
    bottomIdx = getIdx_global(pos2d.x, pos2d.y + 1);
    if (bottomIdx >= 0) {
      counter += inputGrid[bottomIdx] ? 1 : 0;
    }
  }

  // Bottom right
  int bottomRightIdx;
  if (useShared(s_pos2d.x + 1, s_pos2d.y + 1)) {
    bottomRightIdx = getIdx_shared(s_pos2d.x + 1, s_pos2d.y + 1);
    counter += s_inputGrid[bottomRightIdx] ? 1 : 0;
  } else {
    bottomRightIdx = getIdx_global(pos2d.x + 1, pos2d.y + 1);
    if (bottomRightIdx >= 0) {
      counter += inputGrid[bottomRightIdx] ? 1 : 0;
    }
  }

  // Assess counter value
  outputGrid[idx] = assessNeighborCount(counter, s_inputGrid[s_idx]);
}

void initBoard(bool* grid) {
  // Allocate memory and initialize to all 0's.
  memset(grid, 0, sizeof(bool) * GRID_SIZE * GRID_SIZE);

  // Random seed
  srand(time(NULL));

  for (int i = 0; i < GRID_SIZE; ++i) {
    for (int j = 0; j < GRID_SIZE; ++j) {
      // Get 1D mapping
      int idx = i * GRID_SIZE + j;
      bool value = (rand() % 100) > 75;
      grid[idx] = value;
    }
  }
}

void printGrid(bool* grid) {
  printf("---------------\n");
  for (int i = 0; i < GRID_SIZE; ++i) {
    for (int j = 0; j < GRID_SIZE; ++j) {
      int idx = i * GRID_SIZE + j;
      char*  val;
      if (grid[idx]) {
        val = "#";
      } else {
        val = "-";
      }
      std::cout << val << " ";
    }
    std::cout << "\n";
  }
}

void handleSignal(int sigNum) {
  continueNextGeneration = false;
}

int main(int argc, char** argv) {
  const dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);
  const dim3 gridSize(ceil(1.0f*GRID_SIZE / blockSize.x), ceil(1.0f*GRID_SIZE / blockSize.y));
  gpuErrchk(cudaFree(0));

  size_t allocSize = sizeof(bool) * GRID_SIZE * GRID_SIZE;
  bool* d_gridInput;
  bool* d_gridOutput;
  gpuErrchk(cudaMalloc(&d_gridInput, allocSize));
  gpuErrchk(cudaMalloc(&d_gridOutput, allocSize));

  bool* initialGrid = (bool*)malloc(allocSize);
  initBoard(initialGrid);
  gpuErrchk(cudaMemcpy(d_gridInput, initialGrid, allocSize, cudaMemcpyHostToDevice));
  
  printGrid(initialGrid);
  free(initialGrid);

  signal(SIGINT, handleSignal);

  while (continueNextGeneration) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    computeNextGeneration<<<gridSize, blockSize>>>(d_gridInput, d_gridOutput);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    bool* output = (bool*)malloc(allocSize);
    gpuErrchk(cudaMemcpy(output, d_gridOutput, allocSize, cudaMemcpyDeviceToHost));
    printGrid(output);

    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Process time: %f milliseconds.\n", milliseconds);

    gpuErrchk(cudaMemcpy(d_gridInput, d_gridOutput, allocSize, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemset(d_gridOutput, 0, allocSize));

    free(output);
    struct timespec tim, tim2;
    tim.tv_sec = 0;
    tim.tv_nsec = 75000000L;
    nanosleep(&tim, &tim2);
  }

  gpuErrchk(cudaFree(d_gridInput));
  gpuErrchk(cudaFree(d_gridOutput));

  printf("\nExiting.\n");
}