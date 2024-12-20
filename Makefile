CXX := g++
HIPXX := hipcc
CXXFLAGS := -std=c++17 -O3
HIPFLAGS := $(CXXFLAGS) -I /opt/rocm-6.2.1/include/hipblas -L /opt/rocm-6.2.1/lib -l hipblas

.PHONY: para
para: para.cpp matrix.cpp matrix.hpp
	$(HIPXX) $(HIPFLAGS) para.cpp matrix.cpp -o para

.PHONY: seq
seq: seq.cpp matrix.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) seq.cpp matrix.cpp -o seq

.PHONY: clean
clean:
	rm -f seq para

.PHONY: test_matrix
test_matrix: matrix.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) matrix.cpp -o test -D __UNIT_TEST__
	./test
	rm test