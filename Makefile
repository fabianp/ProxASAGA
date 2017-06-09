

all:
	g++ -g3 -std=c++11 -fPIC -c proxasaga_atomic.cpp -o proxasaga_atomic.o
	g++ -shared -o libasaga.so proxasaga_atomic.o

clean:
	rm *.o *.so
