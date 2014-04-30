SHELL = /bin/sh

# Compiler Options
CC = colorgcc
CFLAGS = -c -ansi -pedantic -Wall -Wextra
CVFLAGS = `pkg-config --cflags --libs opencv`

# Directories & Files
BIN_DIR = bin
SRC_DIR = src
INT_DIR = intermediate
AVI_FILE = ../Videos/1.avi
#EXE_FILE := $(BIN_DIR)/main.exe
#CPP_FILES := $(wildcard $(SRC_DIR)/*.cpp)
#OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(BIN_DIR)/%.o,$(CPP_FILES))

# Video Time Range
START_TIME = 930
DURATION = 30

#all: $(EXE_FILE)

#$(EXE_FILE): $(OBJ_FILES)
#	$(CC) -o $@ $^ $(CVFLAGS) -lstdc++

#$(OBJ_FILES): | $(BIN_DIR)

#$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
#	$(CC) -o $@ $< $(CFLAGS)


# Create "bin" directory to place .o and .exe files.
$(BIN_DIR):
	mkdir $(BIN_DIR)

# Create "intermediate" directory to place files created/used between programs.
$(INT_DIR):
	mkdir $(INT_DIR)


# Main. Used for testing small programs temporarily.
$(BIN_DIR)/main.exe: $(BIN_DIR)/main.o
	$(CC) -o $@ $^ $(CVFLAGS) -lstdc++

$(BIN_DIR)/main.o: | $(BIN_DIR)

$(BIN_DIR)/main.o: $(SRC_DIR)/main.cpp
	$(CC) -o $@ $< $(CFLAGS)


# Program to extract background model for BG subtraction.
$(BIN_DIR)/extract_bg.exe: $(BIN_DIR)/extract_bg.o $(BIN_DIR)/helper.o
	$(CC) -o $@ $^ $(CVFLAGS) -lstdc++

$(BIN_DIR)/extract_bg.o: | $(BIN_DIR)

$(BIN_DIR)/helper.o: | $(BIN_DIR)

$(BIN_DIR)/extract_bg.o: $(SRC_DIR)/extract_bg.cpp $(SRC_DIR)/helper.h
	$(CC) -o $@ $< $(CFLAGS)

$(BIN_DIR)/helper.o: $(SRC_DIR)/helper.cpp $(SRC_DIR)/helper.h
	$(CC) -o $@ $< $(CFLAGS)


# Program to test extract_bg.exe by doing BG subtraction.
$(BIN_DIR)/test_bg_subtraction.exe: $(BIN_DIR)/test_bg_subtraction.o $(BIN_DIR)/helper.o
	$(CC) -o $@ $^ $(CVFLAGS) -lstdc++

$(BIN_DIR)/test_bg_subtraction.o: | $(BIN_DIR)

$(BIN_DIR)/test_bg_subtraction.o: $(SRC_DIR)/test_bg_subtraction.cpp $(SRC_DIR)/helper.h
	$(CC) -o $@ $< $(CFLAGS)


# Program to extract trajectories from video.
$(BIN_DIR)/extract_traj.exe: $(BIN_DIR)/extract_traj.o $(BIN_DIR)/helper.o $(BIN_DIR)/TrajectoryListManager.o
	$(CC) -o $@ $^ $(CVFLAGS) -lstdc++

$(BIN_DIR)/extract_traj.o: | $(BIN_DIR)

$(BIN_DIR)/TrajectoryListManager.o: | $(BIN_DIR)

$(BIN_DIR)/extract_traj.o: $(SRC_DIR)/extract_traj.cpp $(SRC_DIR)/helper.h
	$(CC) -o $@ $< $(CFLAGS)

$(BIN_DIR)/TrajectoryListManager.o: $(SRC_DIR)/TrajectoryListManager.cpp $(SRC_DIR)/TrajectoryListManager.h $(SRC_DIR)/helper.h
	$(CC) -o $@ $< $(CFLAGS)


# Program to test extract_traj.exe by displaying trajectories.
$(BIN_DIR)/test_traj_extraction.exe: $(BIN_DIR)/test_traj_extraction.o $(BIN_DIR)/helper.o
	$(CC) -o $@ $^ $(CVFLAGS) -lstdc++

$(BIN_DIR)/test_traj_extraction.o: | $(BIN_DIR)

$(BIN_DIR)/test_traj_extraction.o: $(SRC_DIR)/test_traj_extraction.cpp $(SRC_DIR)/helper.h
	$(CC) -o $@ $< $(CFLAGS)


# Program to group trajectories.
$(BIN_DIR)/group_traj.exe: $(BIN_DIR)/group_traj.o $(BIN_DIR)/helper.o $(BIN_DIR)/BronKerbosch.o
	$(CC) -o $@ $^ $(CVFLAGS) -lstdc++ -lm

$(BIN_DIR)/group_traj.o: | $(BIN_DIR)

$(BIN_DIR)/group_traj.o: $(SRC_DIR)/group_traj.cpp $(SRC_DIR)/helper.h
	$(CC) -o $@ $< $(CFLAGS)


# Program to test Bron-Kerbosch.
$(BIN_DIR)/test_maximal_clique.exe: $(BIN_DIR)/test_maximal_clique.o $(BIN_DIR)/BronKerbosch.o
	$(CC) -o $@ $^ $(CVFLAGS) -lstdc++

$(BIN_DIR)/test_maximal_clique.o: | $(BIN_DIR)

$(BIN_DIR)/BronKerbosch.o: | $(BIN_DIR)

$(BIN_DIR)/test_maximal_clique.o: $(SRC_DIR)/test_maximal_clique.cpp
	$(CC) -o $@ $< $(CFLAGS)

$(BIN_DIR)/BronKerbosch.o: $(SRC_DIR)/BronKerbosch.cpp $(SRC_DIR)/BronKerbosch.h
	$(CC) -o $@ $< $(CFLAGS)


# Play video.
$(BIN_DIR)/play_video.exe: $(BIN_DIR)/play_video.o $(BIN_DIR)/helper.o
	$(CC) -o $@ $^ $(CVFLAGS) -lstdc++

$(BIN_DIR)/play_video.o: | $(BIN_DIR)

$(BIN_DIR)/play_video.o: $(SRC_DIR)/play_video.cpp $(SRC_DIR)/helper.h
	$(CC) -o $@ $< $(CFLAGS)


# Useful Commands
.PHONY: clean
clean:
	-rm -rf ./bin

#.PHONY: run
#run: $(EXE_FILE)
#	./$< $(AVI_FILE)

.PHONY: main
main: $(BIN_DIR)/main.exe
	./$< $(AVI_FILE)

.PHONY: extract_bg
extract_bg: | $(INT_DIR)
extract_bg: $(BIN_DIR)/extract_bg.exe
	./$< $(AVI_FILE) $(INT_DIR)/bg.txt

.PHONY: test_bg_subtraction
test_bg_subtraction: $(BIN_DIR)/test_bg_subtraction.exe
	./$< $(AVI_FILE) $(INT_DIR)/bg.txt

.PHONY: extract_traj
extract_traj: $(BIN_DIR)/extract_traj.exe
	./$< $(AVI_FILE) $(INT_DIR)/bg.txt $(INT_DIR)/traj_$(START_TIME)_$(DURATION).txt $(START_TIME) $(DURATION)

.PHONY: test_traj_extraction
test_traj_extraction: $(BIN_DIR)/test_traj_extraction.exe
	./$< $(AVI_FILE) $(INT_DIR)/traj_$(START_TIME)_$(DURATION).txt

.PHONY: group_traj
group_traj: $(BIN_DIR)/group_traj.exe
	./$< $(AVI_FILE) $(INT_DIR)/traj_$(START_TIME)_$(DURATION).txt $(INT_DIR)/traj_$(START_TIME)_$(DURATION).avi

.PHONY: test_maximal_clique
test_maximal_clique: $(BIN_DIR)/test_maximal_clique.exe
	./$<

.PHONY: play_video
play_video: $(BIN_DIR)/play_video.exe
	./$< $(AVI_FILE) $(START_TIME) $(DURATION)
