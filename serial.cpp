#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <cstring>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <chrono>

#define GRID_SIZE 72

bool continueNextGeneration = true;

bool* grid;

void printBoard() {
  std::cout << "----------------" << std::endl;
  for (int i = 0; i < GRID_SIZE; ++i) {
    for (int j = 0; j < GRID_SIZE; ++j) {
      int idx = i * GRID_SIZE + j;
      std::string val;
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

void initBoard() {
  // Allocate memory and initialize to all 0's.
  grid = (bool*)malloc(sizeof(bool) * GRID_SIZE * GRID_SIZE);
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

int getIdx(int i, int j) {
  if (i < 0 || j < 0 || i >= GRID_SIZE || j >= GRID_SIZE) {
    return -1;
  }
  return i * GRID_SIZE + j;
}

bool assessNeighborCount(int counter, bool currentVal) {
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

/**
 * @brief 
 * RULES:
 * 1. Any living cell (1) with less than 2 neighbors or more than 3 neighbors dies (0)
 * 2. Any dead (0) cell with EXACTLY 3 living neighbors is reborn (1)
 */
void nextGeneration() {
  bool* newGrid = (bool*)malloc(sizeof(bool) * GRID_SIZE * GRID_SIZE);
  for (int i = 0; i < GRID_SIZE; ++i) {
    for (int j = 0; j < GRID_SIZE; ++j) {
      int idx = getIdx(i, j);

      int counter = 0;

      int topLeftIdx = getIdx(i - 1, j - 1);
      if (topLeftIdx >= 0) { counter += grid[topLeftIdx] ? 1 : 0; }
      
      int topIdx = getIdx(i - 1, j);
      if (topIdx >= 0) { counter += grid[topIdx] ? 1 : 0; }

      int topRightIdx = getIdx(i - 1, j + 1);
      if (topRightIdx >= 0) { counter += grid[topRightIdx] ? 1 : 0; }

      int leftIdx = getIdx(i, j - 1);
      if (leftIdx >= 0) { counter += grid[leftIdx] ? 1 : 0; }

      int bottomLeftIdx = getIdx(i + 1, j - 1);
      if (bottomLeftIdx >= 0) { counter += grid[bottomLeftIdx] ? 1 : 0; }

      int bottomIdx = getIdx(i + 1, j);
      if (bottomIdx >= 0) { counter += grid[bottomIdx] ? 1 : 0; }

      int bottomRightIdx = getIdx(i + 1, j + 1);
      if (bottomRightIdx >= 0) { counter += grid[bottomRightIdx] ? 1 : 0; }

      int rightIdx = getIdx(i, j + 1);
      if (rightIdx >= 0) { counter += grid[rightIdx] ? 1 : 0; }

      bool current = grid[idx];
      newGrid[idx] = assessNeighborCount(counter, current);
    }
  }
  memcpy(grid, newGrid, sizeof(bool) * GRID_SIZE * GRID_SIZE);
  free(newGrid);
}

void flipRandomBit() {
  int idx = rand() % (GRID_SIZE * GRID_SIZE) - 1;
  grid[idx] = true;
}

void handleSignal(int sigNum) {
  continueNextGeneration = false;
}

int main() {

  using std::chrono::high_resolution_clock;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  initBoard();
  printBoard();

  signal(SIGINT, handleSignal);
  while (continueNextGeneration) {
    auto start = high_resolution_clock::now();
    nextGeneration();
    auto stop = high_resolution_clock::now();

    duration<double, std::milli> t = stop - start;

    printBoard();
    std::cout << "Process time: " << t.count() << " milliseconds\n";
    
    if ((rand() % 100) > 90) {
      flipRandomBit();
    }

    // sleep(1);

    struct timespec tim, tim2;
    tim.tv_sec = 0;
    tim.tv_nsec = 75000000L;
    nanosleep(&tim, &tim2);
  }

  free(grid);
  return 0;
}