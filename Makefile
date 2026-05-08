CXX = clang++
TARGET = main
SRCS = $(wildcard NN/src/*.cpp)
INCLUDES = -I NN/include
LIBS = -L NN/lib -lraylib


$(TARGET): $(SRCS)
	$(CXX) $(SRCS) $(INCLUDES) $(LIBS) -o $(TARGET)

run:
	./main

clean:
	rm -f $(TARGET)
	rm -f *.out