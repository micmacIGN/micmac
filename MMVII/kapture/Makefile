SOURCES=$(wildcard *.cpp)
CXXFLAGS += -std=c++17 -Wextra -Wall -Werror
LD=g++

ifdef DEBUG
  LIB=libkapture_debug.a
  PGM=kpttest_debug
  CXXFLAGS += -g
  OBJDIR=debug/
else
  LIB=libkapture.a
  PGM=kpttest
  TARGET=libkapture.a kpttest
  CXXFLAGS += -O3
  OBJDIR=release/
endif

OBJS=$(addprefix $(OBJDIR),$(filter-out kpt_test.o,$(SOURCES:%.cpp=%.o)))
INCLUDES +=

.phony: all clean


all: $(LIB) $(PGM)

$(LIB): $(OBJS)
	ar crs $@ $^

$(PGM): $(OBJDIR)kpt_test.o $(LIB)
	$(LD) $(LDFLAGS) -o $@ $^


clean:
	rm -f $(PGM) $(LIB)
	rm -fr $(OBJDIR)

cleanall:
	rm -f  *.a kpttest kpttest_debug
	rm -fr release debug

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(OBJDIR)%.o: %.cpp Makefile | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<


$(OBJDIR).%.d: %.cpp
	@set -e; rm -f $@; \
	$(CXX) -MM -MP -MT '$(OBJDIR)$(patsubst %.cpp,%.o,$<) $(OBJDIR)$(patsubst %.cpp,%.d,$<)' -MF $@ $(CXXFLAGS) $(INCLUDES) $<   2>/dev/null

ifneq ($(MAKECMDGOALS),clean)
-include $(SOURCES:%.cpp=$(OBJDIR).%.d)
endif

