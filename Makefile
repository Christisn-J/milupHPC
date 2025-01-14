#Compiler/Linker
CXX            := mpic++

CUDADIR        := /usr/local/cuda-11.4
NVCC           := $(CUDADIR)/bin/nvcc

OPENMPIDIR     := /opt/openmpi-4.1.0
HDF5DIR        := /usr/include/hdf5/openmpi
HDF5LIB        := /usr/lib/x86_64-linux-gnu/hdf5/openmpi
HEADERONLYDIR  := /usr/local/Headeronly

#Target binary
TARGET         := runner

#Directories
SRCDIR         := ./src
INCDIR         := ./include
BUILDDIR       := ./build
TARGETDIR      := ./bin
RESDIR         := ./resources
# IDEASDIR       := ./ideas
# TESTDIR        := ./test
DOCDIR         := ./doc
DOCUMENTSDIR   := ./documents

SRCEXT         := cpp
CUDASRCEXT     := cu
DEPEXT         := d
OBJEXT         := o

#Flags, Libraries and Includes
CXXFLAGS       += -std=c++11 -w -I/usr/include/hdf5/openmpi#-O3
NVFLAGS        := --std=c++11 -x cu -c -dc -w -Xcompiler "-pthread" -Wno-deprecated-gpu-targets -fmad=false -O3 -I$(OPENMPIDIR)/include -I$(HDF5DIR)
LFLAGS         += -lm -L$(CUDADIR)/lib64 -lcudart -lpthread -lconfig -L$(OPENMPIDIR)/lib -L$(HDF5LIB) -lmpi -lhdf5 -lboost_filesystem -lboost_system
GPU_ARCH       := -arch=sm_52
CUDALFLAGS     := -dlink
CUDALINKOBJ    := cuLink.o #needed?
LIB            := -lboost_mpi -lboost_serialization
INC            := -I$(INCDIR) -I/usr/include -I$(CUDADIR)/include -I$(OPENMPIDIR)/include -I$(HEADERONLYDIR)
INCDEP         := -I$(INCDIR)

#Source and Object files
#find ./src -type f -name "*.cu" -not -path "./src/gravity/*"
#find ./src -type f -name "*.cu" -not -path "*/gravity/*"
#find . -type d \( -path ./src/sph -o -path ./src/gravity -o -path ./dir3 \) -prune -o -name '*.cu' -print
#find . -type d \( -name sph -o -name gravity -o -name dir3 \) -prune -o -name '*.cu' -print
SOURCES        := $(shell find $(SRCDIR) -type f -name "*.$(SRCEXT)")
CUDA_SOURCES   := $(shell find $(SRCDIR) -type f -name "*.$(CUDASRCEXT)")
OBJECTS        := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))
CUDA_OBJECTS   := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CUDA_SOURCES:.$(CUDASRCEXT)=.$(OBJEXT)))

#Documentation (Doxygen)
DOXY           := /usr/local/Cellar/doxygen/1.9.3_1/bin/doxygen
DOXYFILE       := $(DOCDIR)/Doxyfile

#default make (all)
# all:  tester ideas $(TARGET)
all: $(TARGET)

single-precision: CXXFLAGS += -DSINGLE_PRECISION
single-precision: NVFLAGS += -DSINGLE_PRECISION
single-precision: all

debug: CXXFLAGS += -g
debug: NVFLAGS := ${filter-out -O3, $(NVFLAGS)}
debug: NVFLAGS  += -g -G
debug: LFALGS += -g -G
debug: tester ideas $(TARGET)

#make regarding source files
sources: resources $(TARGET)

#remake
remake: cleaner all

#copy Resources from Resources Directory to Target Directory
resources: directories
	@cp -r $(RESDIR)/ $(TARGETDIR)/

#make directories
directories:
	@mkdir -p $(RESDIR)
	@mkdir -p $(TARGETDIR)
	@mkdir -p $(BUILDDIR)

#clean objects
clean:
	@$(RM) -rf $(BUILDDIR)

#clean objects and binaries
cleaner: clean
	@$(RM) -rf $(TARGETDIR)

#Pull in dependency info for *existing* .o files
-include $(OBJECTS:.$(OBJEXT)=.$(DEPEXT)) #$(INCDIR)/matplotlibcpp.h

#link
$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	@echo "Linking ..."
	@mkdir -p $(TARGETDIR)
	@$(NVCC) $(GPU_ARCH) $(LFLAGS) $(INC) -o $(TARGETDIR)/$(TARGET) $^ $(LIB) #$(GPU_ARCH)

#compile
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@echo "  compiling: " $(SRCDIR)/$*
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INC) -c -o $@ $< $(LIB)
	@$(CXX) $(CXXFLAGS) $(INC) $(INCDEP) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp

$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(CUDASRCEXT)
	@echo "  compiling: " $(SRCDIR)/$*
	@mkdir -p $(dir $@)
	@$(NVCC) $(GPU_ARCH) $(INC) $(NVFLAGS) -I$(CUDADIR) -c -o $@ $<
	@$(NVCC) $(GPU_ARCH) $(INC) $(NVFLAGS) -I$(CUDADIR) -MM $(SRCDIR)/$*.$(CUDASRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)

#compile test files
# tester: directories
# ifneq ("$(wildcard $(TESTDIR)/*.$(SRCEXT) )","")
#  	@echo "  compiling: " test/*
# 	@$(CXX) $(CXXFLAGS) test/*.cpp $(INC) $(LIB) -o bin/tester
# else
# 	@echo "No $(SRCEXT)-files within $(TESTDIR)!"
# endif


#compile idea files
# ideas: directories
# ifneq ("$(wildcard $(IDEASDIR)/*.$(SRCEXT) )","")
# 	@echo "  compiling: " ideas/*
# 	@$(CXX) $(CXXFLAGS) ideas/*.cpp $(INC) $(LIB) -o bin/ideas
# else
# 	@echo "No $(SRCEXT)-files within $(IDEASDIR)!"
# endif

#@echo FILE_PATTERNS     = "*.md" "*.h" "*.$(SRCEXT)" >> $(DOCDIR)/doxyfile.inc
doxyfile.inc: #Makefile
	@echo INPUT            = README.md . $(SRCDIR)/ $(INCDIR)/ $(DOCUMENTSDIR)/ > $(DOCDIR)/doxyfile.inc
	@echo OUTPUT_DIRECTORY = $(DOCDIR)/ >> $(DOCDIR)/doxyfile.inc

#@$(MAKE) -C $(DOCDIR)/latex/ &> $(DOCDIR)/latex/latex.log
doc: doxyfile.inc
	$(DOXY) $(DOXYFILE) &> $(DOCDIR)/doxygen.log
	@mkdir -p "./docs"
	cp -r "./doc/html/" "./docs/"
	@cp -r "./documents" "./docs/"
	@cp "./doc/open.png" "./docs/"
	@cp "./doc/closed.png" "./docs/"

remove_doc:
	rm -rf docs/*
	rm -rf doc/html

#Non-File Targets
.PHONY: all remake clean cleaner resources sources directories ideas tester doc remove_doc
