INCDIR=-I ../../ -I../../include
CXXFLAGS=-fopenmp -std=c++17 -Wall -Werror -O4 -march=native $(INCDIR) -D"SYMBDER_WITH_MMVII=false"
SYMBDER_INCLUDES=$(wildcard $(INCDIR)/SymbDer/*.h)

all: CodeGen_IncludeAll.h TestCodeGen


CodeGenerator: CodeGenerator.cpp Formula_*.h $(SYMBDER_INCLUDES)
	$(CXX) $< $(CXXFLAGS)  -o $@

CodeGen_IncludeAll.h: CodeGenerator
	@echo "* Generating Formulas code"
	./CodeGenerator
	$(MAKE) TestCodeGen

FUNCTIONS=$(patsubst %.cpp,%.o,$(wildcard CodeGen_*.cpp))

CodeGen_%.o: CodeGen_%.cpp CodeGen_%.h
	$(CXX) -c $< $(CXXFLAGS)  -o $@

TestCodeGen: TestCodeGen.cpp $(FUNCTIONS) TestCodeGenTpl.h ChronoBench.h CodeGen_IncludeAll.h $(SYMBDER_INCLUDES)
	$(CXX) $< $(FUNCTIONS) $(CXXFLAGS)  -o $@
 
clean:
	rm -f *.o CodeGen_* CodeGenerator TestCodeGen
