SOURCES=$(wildcard *.cpp)
CXXFLAGS += -std=c++17 -Wextra -Wall -Werror
LD=g++

ifdef DEBUG
  LIB=libkapture_debug.a
  CXXFLAGS += -g
  OBJDIR=debug/
else
  LIB=libkapture.a
  TARGET=libkapture.a kpttest
  CXXFLAGS += -O3
  OBJDIR=release/
endif

OBJS:=$(filter-out kpt_mmv2.o,$(SOURCES:%.cpp=%.o))

OBJS:=$(addprefix $(OBJDIR),$(OBJS))
INCLUDES +=

.phony: all clean


all: $(LIB)

$(LIB): $(OBJS)
	ar crs $@ $^


clean:
	rm -f $(LIB) *.o
	rm -fr $(OBJDIR)

cleanall:
	rm -f *.o *.a 
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

