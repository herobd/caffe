CXX = g++ 
LINK = g++
#CXX_FLAGS =  -fPIC  -std=c++11  -DOPENCV2 -O3 #-DNO_FEAT -O3  #-g
CXX_FLAGS =  -fPIC  -std=c++11 -g#-O3#-DSAVE_IMAGES -g #-DNO_FEAT#-O3  #-g

SOURCES += main.cpp


#LIBS +=  -pthread -lglog -l:libopencv_core.so -l:libopencv_imgproc.so -l:libopencv_highgui.so -lprotobuf -lboost_system
#LIBS +=  -lglog -l:libopencv_core.so.3.0 -l:libopencv_imgcodecs.so.3.0 -l:libopencv_imgproc.so.3.0 -l:libopencv_highgui.so.3.0 -lprotobuf -lboost_system
LIBS +=  -lglog -l:libopencv_core.so.3.1 -l:libopencv_imgcodecs.so.3.1 -l:libopencv_imgproc.so.3.1 -l:libopencv_highgui.so.3.1 -lprotobuf -lboost_system
#LIBS +=  -lglog -l:libopencv_core.so.3.1 -l:libopencv_imgcodecs.so.3.1 -l:libopencv_imgproc.so.3.1 -l:libopencv_highgui.so.3.1 -lprotobuf -l:libboost_system.so.1.59.0
LIBS += -L ../../build/lib/ -lcaffe #caffe
INCLUDE += -I ../../include/ #caffe



PROGRAM_NAME = cnnspp_spotter
LIB_NAME = libcnnspp_spotter.a

OBJECTS = obj/main.o obj/cnnspp_spotter.o obj/gwdataset.o obj/cnn_featurizer.o obj/spp_embedder.o obj/phocer.o

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

obj/cnnspp_spotter.o: cnnspp_spotter.h cnnspp_spotter.cpp cnnspp_spotter_eval.cpp dataset.h SubwordSpottingResult.h cnn_featurizer.h spp_embedder.h phocer.h Transcriber.h
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/cnnspp_spotter.o cnnspp_spotter.cpp $(LIBS)
	
obj/gwdataset.o: dataset.h gwdataset.h gwdataset.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/gwdataset.o gwdataset.cpp

obj/cnn_featurizer.o: cnn_featurizer.h cnn_featurizer.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/cnn_featurizer.o cnn_featurizer.cpp

obj/spp_embedder.o: spp_embedder.h spp_embedder.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/spp_embedder.o spp_embedder.cpp

obj/phocer.o: phocer.h phocer.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/phocer.o phocer.cpp

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../build/lib
