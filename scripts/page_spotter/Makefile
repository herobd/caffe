CXX = g++ 
LINK = g++
#CXX_FLAGS =  -fPIC  -std=c++11  -DOPENCV2 -O3 
CXX_FLAGS =  -fPIC  -std=c++11 -g#-DSHOW  

SOURCES += main.cpp


#LIBS +=  -pthread -lglog -l:libopencv_core.so -l:libopencv_imgproc.so -l:libopencv_highgui.so -lprotobuf -lboost_system
#LIBS +=  -lglog -l:libopencv_core.so.3.0 -l:libopencv_imgcodecs.so.3.0 -l:libopencv_imgproc.so.3.0 -l:libopencv_highgui.so.3.0 -lprotobuf -lboost_system
LIBS +=  -lglog -l:libopencv_core.so.3.1 -l:libopencv_imgcodecs.so.3.1 -l:libopencv_imgproc.so.3.1 -l:libopencv_highgui.so.3.1 -lprotobuf -lboost_system
LIBS += -L ../../build/lib/ -lcaffe #caffe
INCLUDE += -I ../../include/ #caffe

#g++ -std=c++11 -fopenmp gwdataset.cpp evalPageSpotting.cpp -lcaffe -lglog -l:libopencv_core.so.3.1 -l:libopencv_imgcodecs.so.3.1 -l:libopencv_imgproc.so.3.1 -l:libopencv_highgui.so.3.1 -lprotobuf -lboost_system -I ../include/ -L ../build/lib/ -o evalPageSpotting

PROGRAM_NAME = evalPageSpotting
LIB_NAME = libpage_spotter.a

OBJECTS = obj/evalPageSpotting.o obj/cnn_featurizer.o obj/cnn_spotter.o obj/BBPredictor.o

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
	#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../build/lib
	$(LINK) $(CXX_FLAGS) -o $(PROGRAM_NAME) $(OBJECTS) $(LIBS) 

	
obj/evalPageSpotting.o: evalPageSpotting.cpp cnn_featurizer.h cnn_spotter.h
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/evalPageSpotting.o evalPageSpotting.cpp $(LIBS)


obj/cnn_featurizer.o: cnn_featurizer.h cnn_featurizer.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/cnn_featurizer.o cnn_featurizer.cpp

obj/cnn_spotter.o: cnn_spotter.h cnn_spotter.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/cnn_spotter.o cnn_spotter.cpp

obj/BBPredictor.o: BBPredictor.h BBPredictor.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE) -o obj/BBPredictor.o BBPredictor.cpp


