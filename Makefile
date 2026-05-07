CXX = clang++
TARGET = main
SRCS = $(wildcard NN/src/*.cpp)

$(TARGET): $(SRCS)
	$(CXX) $(SRCS) -o $(TARGET)

run:
	./main

clean:
	rm -f $(TARGET)