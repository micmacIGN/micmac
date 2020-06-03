INCDIR=../../include/
CXXFLAGS=-fopenmp -std=c++14 -Wall -Werror -O4 -march=native -I$(INCDIR)
SYMBDER_INCLUDES=$(wildcard $(INCDIR)/SymbDer/*.h)

all: TestCodeGen


CodeGenerator: CodeGenerator.cpp Formula_*.h $(SYMBDER_INCLUDES)
	$(CXX) $< $(CXXFLAGS)  -o $@

CodeGen_IncludeAll.h: CodeGenerator
	@echo "* Generating Formulas code"
	./CodeGenerator

TestCodeGen: TestCodeGen.cpp TestCodeGenTpl.h ChronoBench.h CodeGen_IncludeAll.h $(SYMBDER_INCLUDES)
	$(CXX) $< $(CXXFLAGS)  -o $@
 
clean:
	rm -f *.o CodeGen*.h CodeGenerator TestCodeGen
