CXX = g++ 
LINK = g++
CXX_FLAGS =  -fPIC  -std=c++11 -g#-O3  #-g

SOURCES += main.cpp


LIBS +=  -lcaffe -lglog -l:libopencv_core.so.3.0 -l:libopencv_imgcodecs.so.3.0 -l:libopencv_imgproc.so.3.0 -l:libopencv_highgui.so.3.0 -lprotobuf -lboost_system
LIBS += -L ../../build/lib/ #caffe
INCLUDE += -I ../../include/ #caffe



PROGRAM_NAME = cnnspotter
LIB_NAME = libcnnspotter.a

OBJECTS = obj/main.o obj/cnnspotter.o obj/gwdataset.o obj/cnnembedder.o

bin: $(PROGRAM_NAME)
all: $(PROGRAM_NAME) $(LIB_NAME)
lib: $(LIB_NAME)

clean:
	- rm  $(PROGRAM_NAME)
	- rm  $(LIB_NAME)
	- rm  obj/*.o

$(LIB_NAME): $(OBJECTS)
	- rm  $(LIB_NAME)
	ar -cvq $(LIB_NAME) $(OBJECTS)

$(PROGRAM_NAME):  $(OBJECTS)  
	$(LINK) $(CXX_FLAGS) -o $(PROGRAM_NAME) $(OBJECTS) $(LIBS) 

	
obj/main.o: main.cpp gwdataset.h
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/main.o main.cpp $(LIBS)

obj/cnnspotter.o: cnnspotter.h cnnspotter.cpp cnnspotter_eval.cpp gwdataset.h SubwordSpottingResult.h
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/cnnspotter.o cnnspotter.cpp $(LIBS)
	
obj/gwdataset.o: dataset.h gwdataset.h gwdataset.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/gwdataset.o gwdataset.cpp

obj/cnnembedder.o: cnnembedder.h cnnembedder.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/cnnembedder.o cnnembedder.cpp


